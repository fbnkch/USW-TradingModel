"""
Account-Manager: Positions-Tracking, Risk-Management, Exit-Logik.

Regeln:
  - Max. 10 gleichzeitige Positionen
  - Max. 0.5% Risk pro Trade
  - Take Profit: +0.25%
  - Stop Loss: -1.0% (fix, weiter Stop fuers Atmen)
  - Trailing Stop Loss (opt-in): graduell, SL wandert mit steigendem Profit nach oben
  - Time Stop: 30 Minuten
  - Signal-Kollaps: Finder-Score < 0.20
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class Position:
    """Tracking-Dataclass fuer eine offene Position."""
    symbol: str
    entry_time: datetime
    entry_price: float
    qty: int
    tp_price: float
    sl_price: float               # initialer Stop Loss (absolutes Minimum)
    highest_price: float = 0.0    # hoechster erreichte Preis seit Entry (fuer Trailing SL)
    ratchet_triggered: bool = False  # True wenn TP-Level einmal erreicht wurde
    ratchet_floor: float = 0.0    # neue SL-Unterkante nach Ratchet (≥ TP-Level)
    time_stop_minutes: int = 30   # individueller Time-Stop (Regime-abhaengig)
    order_id: str = ""
    current_finder_score: float = 0.0


@dataclass
class ExitAction:
    """Beschreibt eine auszufuehrende Exit-Order."""
    symbol: str
    reason: str  # "take_profit", "ratchet_exit", "trailing_stop", "stop_loss", "time_stop", "signal_collapse"
    entry_price: float
    current_price: float
    pnl_pct: float


class AccountManager:
    """Verwaltet Positionen und Risk-Management."""

    def __init__(
        self,
        trading_client,
        max_positions: int = 10,
        max_risk_per_trade: float = 0.005,
        tp_pct: float = 0.0035,
        sl_pct: float = 0.0060,
        time_stop_minutes: int = 30,
        signal_collapse_threshold: float = 0.20,
        position_size_pct: float = 0.05,
        trailing_sl_pct: float = 0.004,
        trailing_min_pct: float = 0.0015,
        trailing_ramp_pct: float = 0.004,
        enable_trailing_sl: bool = True,
        ratchet_mode: bool = True,
        reentry_cooldown_minutes: int = 5,
        sl_time_decay_target: float = 0.003,
        sl_time_decay_grace: int = 5,
        grace_sl_pct: float = 0.015,
        grace_sl_pct_t2: float = 0.010,
        entry_grace_minutes: int = 3,
        entry_grace_t2_minutes: int = 5,
        logger=None,
    ):
        self._client = trading_client
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.time_stop = timedelta(minutes=time_stop_minutes)
        self.time_stop_minutes = time_stop_minutes
        self.signal_collapse_threshold = signal_collapse_threshold
        self.position_size_pct = position_size_pct
        self.trailing_sl_pct = trailing_sl_pct
        self.trailing_min_pct = trailing_min_pct
        self.trailing_ramp_pct = trailing_ramp_pct
        self.enable_trailing_sl = enable_trailing_sl
        self.ratchet_mode = ratchet_mode
        self.reentry_cooldown = timedelta(minutes=reentry_cooldown_minutes)
        self.sl_time_decay_target = sl_time_decay_target
        self.sl_time_decay_grace = sl_time_decay_grace
        self.grace_sl_pct = grace_sl_pct
        self.grace_sl_pct_t2 = grace_sl_pct_t2
        self.entry_grace_minutes = entry_grace_minutes
        self.entry_grace_t2_minutes = entry_grace_t2_minutes
        self.logger = logger

        self._positions: dict[str, Position] = {}
        self._symbol_traded_today: set[str] = set()
        self._last_exit_time: dict[str, datetime] = {}  # Wash-Trade-Schutz

    # ---- Properties ---------------------------------------------------

    @property
    def open_positions(self) -> int:
        return len(self._positions)

    @property
    def open_symbols(self) -> list[str]:
        return list(self._positions.keys())

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def set_cooldown(self, minutes: int):
        """Aktualisiert den Re-Entry-Cooldown (Regime-abhaengig)."""
        self.reentry_cooldown = timedelta(minutes=minutes)

    # ---- Position Sizing ----------------------------------------------

    def calculate_size(self, equity: float, price: float) -> int:
        """Berechnet Positionsgroesse als Anteil des Equities.

        Beispiel: $100.000 Equity, position_size_pct=5% → $5.000 pro Position.
        Bei max_positions=10 sind maximal 50% des Kapitals gleichzeitig investiert.
        """
        alloc = equity * self.position_size_pct
        qty = int(alloc / price)
        return max(1, qty)

    # ---- Entry --------------------------------------------------------

    def can_enter(self, symbol: str) -> bool:
        """Prueft ob ein neuer Trade eroeffnet werden darf.

        Regeln:
          1. Max. gleichzeitige Positionen nicht ueberschritten
          2. Kein Doppel-Einstieg ins gleiche Symbol
          3. Wash-Trade-Schutz: 2 Min Sperre nach Exit (Alpaca blockt Gegenorders)
        """
        if self.open_positions >= self.max_positions:
            return False
        if symbol in self._positions:
            return False  # Kein Doppel-Einstieg ins gleiche Symbol

        # Wash-Trade-Cooldown: Alpaca lehnt Orders ab, wenn innerhalb ~30s
        # eine Gegenorder (Exit) fuers gleiche Symbol ausgefuehrt wurde.
        last_exit = self._last_exit_time.get(symbol)
        if last_exit is not None:
            if datetime.now(timezone.utc) - last_exit < self.reentry_cooldown:
                return False

        return True

    def register_entry(
        self,
        symbol: str,
        entry_price: float,
        qty: int,
        order_id: str = "",
        finder_score: float = 0.0,
        time_stop_minutes: int = None,
    ):
        """Registriert eine neue Position NACH erfolgreicher Order-Fill."""
        tp = entry_price * (1.0 + self.tp_pct)
        sl = entry_price * (1.0 - self.sl_pct)

        self._positions[symbol] = Position(
            symbol=symbol,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            qty=qty,
            tp_price=tp,
            sl_price=sl,
            highest_price=entry_price,  # Startwert = Entry-Preis
            time_stop_minutes=time_stop_minutes or self.time_stop_minutes,
            order_id=order_id,
            current_finder_score=finder_score,
        )
        self._symbol_traded_today.add(symbol)

        if self.logger:
            self.logger.log_order(
                order_id=order_id,
                symbol=symbol,
                side="BUY",
                qty=qty,
                order_type="MARKET",
                status="FILLED",
                filled_price=entry_price,
                filled_at=datetime.now(timezone.utc).isoformat(),
            )

    # ---- Exit Checks --------------------------------------------------

    def update_finder_scores(self, scores: dict[str, float]):
        """Aktualisiert Finder-Scores fuer Signal-Kollaps-Check."""
        for sym, score in scores.items():
            if sym in self._positions:
                self._positions[sym].current_finder_score = score

    def _current_trail_distance(self, pos: Position) -> float:
        """Berechnet den Trail-Abstand linear basierend auf dem maximal erreichten Profit.

        Je weiter der Kurs ueber Entry gestiegen ist, desto enger wird der Trail:
          - Bei Entry (Profit=0):    Trail = trailing_sl_pct  (z.B. 0.15%)
          - Bei voller Rampe:        Trail = trailing_min_pct (z.B. 0.05%)

        Der Trail-Abstand schrumpft linear mit dem highest_price.
        Da highest_price nur steigt, kann der Trail NIE wieder weiter werden.
        """
        profit_at_peak = (pos.highest_price - pos.entry_price) / pos.entry_price

        if profit_at_peak <= 0:
            return self.trailing_sl_pct

        # Lineare Interpolation: 0% Profit → trailing_sl_pct, ramp_pct Profit → trailing_min_pct
        ramp = min(1.0, profit_at_peak / self.trailing_ramp_pct)
        trail = self.trailing_sl_pct + (self.trailing_min_pct - self.trailing_sl_pct) * ramp

        return max(self.trailing_min_pct, trail)

    def check_exits(self, current_prices: dict[str, float]) -> list[ExitAction]:
        """Prueft alle offenen Positionen auf TP/SL/Time-Stop/Signal-Kollaps.

        Vier-Schicht-Schutzsystem (jede Schicht kann den SL nur ANHEBEN):

        1. Fixer SL (sl_pct):            Absoluter Boden bei Entry (−1.0%)
        2. Time-Decay SL:                Zieht sich linear mit der Zeit zusammen
           Min 0: −1.0% → Min 30: −0.3%  (verhindert Time-Stop-Todesfaelle)
        3. Trailing Stop (preisbasiert):  Wandert mit steigendem Kurs nach oben
           Schläft bis Kurs > Entry. Trail schrumpft graduell (0.50→0.20%).
        4. Ratchet (gewinnbasiert):       TP erreicht → SL springt auf TP-Niveau
           Danach nie mehr unter TP fallen.

        Effektiver SL = max(alle vier Schichten). Kann NUR steigen, NIE fallen.

        Returns:
            Liste von ExitActions fuer Positionen die geschlossen werden muessen.
        """
        now = datetime.now(timezone.utc)
        exits = []

        for sym, pos in list(self._positions.items()):
            if sym not in current_prices:
                continue

            price = current_prices[sym]
            age = now - pos.entry_time
            pnl = (price - pos.entry_price) / pos.entry_price

            # --- Trailing Stop: Hoechstpreis aktualisieren ---
            if price > pos.highest_price:
                pos.highest_price = price

            # --- Ratchet-Mode: TP erreicht → SL auf TP-Niveau anheben ---
            if self.ratchet_mode and not pos.ratchet_triggered and price >= pos.tp_price:
                pos.ratchet_triggered = True
                pos.ratchet_floor = pos.tp_price  # Gewinn gesperrt – nie mehr unter TP fallen
                if self.logger:
                    print(f"  🔒 RATCHET: {sym} TP-Level erreicht! "
                          f"SL-Boden jetzt bei ${pos.ratchet_floor:.2f} "
                          f"(+{(pos.ratchet_floor - pos.entry_price) / pos.entry_price * 100:.2f}%)")

            # --- Time-Decay Stop Loss: SL zieht sich mit der Zeit zusammen ---
            # Gnadenfrist: Erste N Minuten volle Atemluft (sl_pct), dann linearer Decay
            # auf sl_time_decay_target bei Time-Stop.
            # Verhindert Time-Stop-Todesfaelle, ohne fruehe Trades zu ersticken.
            age_minutes = age.total_seconds() / 60.0

            # --- Entry-Grace-Period: Gestufter SL in den ersten Minuten ---
            # Daten vom 08.07.: 11 Trades in 0-5 Min, alle Verluste, -$247.
            # Der normale SL (0.60%) gibt bei 1-Min-Bars + Market-Slippage nicht
            # genug Atemluft. Gestufte Grace: nur echte Abstuerze brechen frueh ab.
            #
            # Stufe 1 (0-3 Min): SL 1.50% — nur echte Crashs (>1.5% unter Entry)
            # Stufe 2 (3-5 Min): SL 1.00% — atmen, aber nicht unbegrenzt
            # Normal  (5+  Min): SL 0.60% — normale Disziplin
            #
            # BUGFIX (08.07. 18:00): Waehrend der Grace-Phase wird der Time-Decay-SL
            # deaktiviert. Vorher hat max(base_sl=1.5%, time_sl=0.4%) den engeren
            # Time-Decay genommen — Grace war wirkungslos.
            in_grace = age_minutes < self.entry_grace_t2_minutes
            if age_minutes < self.entry_grace_minutes:
                active_sl_pct = self.grace_sl_pct        # 1.50%
            elif age_minutes < self.entry_grace_t2_minutes:
                active_sl_pct = self.grace_sl_pct_t2     # 1.00%
            else:
                active_sl_pct = self.sl_pct              # 0.60%

            # Time-Decay: Nur AUSSERHALB der Entry-Grace aktiv.
            # Waehrend der Grace-Phase (0-5 Min) gibt es KEINEN Time-Decay —
            # die Grace-Stufen sind der einzige Schutz. Verhindert dass der
            # engere Time-Decay-SL den Grace-Schutz ueberschreibt (Bug 18:00).
            pos_time_stop = float(getattr(pos, 'time_stop_minutes', self.time_stop_minutes))
            if in_grace:
                time_sl = 0.0  # Kein Time-Decay waehrend Grace
            else:
                grace_decay = float(self.sl_time_decay_grace)
                remaining = pos_time_stop - grace_decay
                if remaining > 0 and age_minutes > grace_decay:
                    time_ratio = min(1.0, (age_minutes - grace_decay) / remaining)
                    decay_sl_pct = self.sl_pct + (self.sl_time_decay_target - self.sl_pct) * time_ratio
                else:
                    decay_sl_pct = self.sl_pct
                time_sl = pos.entry_price * (1.0 - decay_sl_pct)

            base_sl = pos.entry_price * (1.0 - active_sl_pct)

            # Effektiven Stop Loss berechnen (Maximum aller Schutz-Schichten)
            # Trailing greift ERST wenn Kurs mindestens trailing_sl_pct ueber Entry war.
            trailing_active = (
                self.enable_trailing_sl
                and pos.highest_price >= pos.entry_price * (1.0 + self.trailing_sl_pct)
            )
            if trailing_active:
                trail_distance = self._current_trail_distance(pos)
                trailing_sl = pos.highest_price * (1.0 - trail_distance)
                effective_sl = max(base_sl, time_sl, pos.ratchet_floor, trailing_sl)
            else:
                effective_sl = max(base_sl, time_sl, pos.ratchet_floor)

            # Ratchet-Mode Exit (Kurs unter Ratchet-Floor gefallen)
            if self.ratchet_mode and pos.ratchet_triggered and price <= pos.ratchet_floor:
                exits.append(ExitAction(sym, "ratchet_exit", pos.entry_price, price, pnl))

            # Take Profit (nur wenn Ratchet-Mode AUS)
            elif not self.ratchet_mode and price >= pos.tp_price:
                exits.append(ExitAction(sym, "take_profit", pos.entry_price, price, pnl))

            # Trailing Stop Loss (erst aktiv wenn trailing-Schwelle erreicht wurde)
            # BUGFIX (2026-07-07): Trailing Stop nur ueber Entry ausloesen.
            # Vorher: Trailing konnte bei -0.2% ausloesen, weil die 1-Min-Bar
            # ein Hoch ueber +0.4% hatte, aber der Close unter Entry lag.
            # Jetzt: Faellt der Preis unter Entry, greift der normale SL.
            elif (self.enable_trailing_sl
                  and pos.highest_price >= pos.entry_price * (1.0 + self.trailing_sl_pct)
                  and price <= effective_sl
                  and price >= pos.entry_price):  # Profit-Floor: kein Trailing-Exit unter Entry!
                exits.append(ExitAction(sym, "trailing_stop", pos.entry_price, price, pnl))

            # Stop Loss (fix, greift wenn:
            #   - Trailing nie aktiviert wurde, ODER
            #   - Preis unter Entry gefallen ist nach Trailing-Aktivierung)
            elif price <= effective_sl:
                exits.append(ExitAction(sym, "stop_loss", pos.entry_price, price, pnl))

            # Time Stop
            elif age >= timedelta(minutes=pos_time_stop):
                exits.append(ExitAction(sym, "time_stop", pos.entry_price, price, pnl))

            # Signal Collapse (Finder-Score eingebrochen)
            elif pos.current_finder_score < self.signal_collapse_threshold and age > timedelta(minutes=5):
                exits.append(ExitAction(sym, "signal_collapse", pos.entry_price, price, pnl))

        return exits

    def register_exit(self, action: ExitAction):
        """Registriert einen Exit und entfernt die Position."""
        if action.symbol in self._positions:
            pos = self._positions.pop(action.symbol)

            # Wash-Trade-Schutz: Exit-Zeitpunkt merken
            self._last_exit_time[action.symbol] = datetime.now(timezone.utc)

            if self.logger:
                self.logger.log_position(
                    symbol=action.symbol,
                    entry_time=pos.entry_time.isoformat(),
                    entry_price=pos.entry_price,
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    exit_price=action.current_price,
                    pnl=(action.current_price - pos.entry_price) * pos.qty,
                    pnl_pct=action.pnl_pct,
                    exit_reason=action.reason,
                )

    # ---- API-Orders ---------------------------------------------------

    def submit_market_sell(self, symbol: str, qty: int) -> Optional[str]:
        """Sendet eine Market-Sell-Order an Alpaca."""
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )
            submitted = self._client.submit_order(order)

            if self.logger:
                self.logger.log_order(
                    order_id=str(submitted.id),
                    symbol=symbol,
                    side="SELL",
                    qty=qty,
                    order_type="MARKET",
                    status=str(submitted.status),
                )
            return str(submitted.id)
        except Exception as e:
            print(f"[AccountManager] SELL-Order fehlgeschlagen fuer {symbol}: {e}")
            return None

    def submit_market_buy(
        self, symbol: str, qty: int, tp_price: float, sl_price: float
    ) -> tuple[Optional[str], float]:
        """Einfache Market-Buy-Order (KEINE Bracket-Orders).

        TP/SL werden via check_exits() im Minutentakt gemanaged.
        """
        import time as _time

        def _wait_for_fill(order_id: str, timeout: float = 3.0) -> float:
            deadline = _time.time() + timeout
            while _time.time() < deadline:
                try:
                    status = self._client.get_order_by_id(order_id)
                    if status.status.value == "filled":
                        return float(status.filled_avg_price) if status.filled_avg_price else 0.0
                    if status.status.value in ("rejected", "canceled", "expired"):
                        return 0.0
                except Exception:
                    pass
                _time.sleep(0.3)
            return 0.0

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

            order = MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                type=OrderType.MARKET, time_in_force=TimeInForce.DAY,
            )
            submitted = self._client.submit_order(order)
            oid = str(submitted.id)
            fill_price = _wait_for_fill(oid)

            if self.logger:
                self.logger.log_order(
                    order_id=oid, symbol=symbol, side="BUY",
                    qty=qty, order_type="MARKET", status=str(submitted.status),
                )
            return oid, fill_price if fill_price > 0 else 0.0
        except Exception as e:
            print(f"[AccountManager] Buy-Order fehlgeschlagen fuer {symbol}: {e}")
            return None, 0.0

    # ---- Emergency Close All ------------------------------------------

    def close_all_positions(self):
        """Verkauft ALLE offenen Positionen (fuer Shutdown/Crash)."""
        if not self._positions:
            print("[Shutdown] Keine offenen Positionen.")
            return
        print(f"[Shutdown] Verkaufe {len(self._positions)} offene Positionen...")
        for sym, pos in list(self._positions.items()):
            try:
                self.submit_market_sell(sym, pos.qty)
                print(f"  [Shutdown] {sym}: SELL {pos.qty} @ Market")
            except Exception as e:
                print(f"  [Shutdown] {sym}: FEHLER {e}")
        self._positions.clear()
        print("[Shutdown] Alle Positionen verkauft.")

    # ---- Daily Reset --------------------------------------------------

    def reset_daily(self):
        """Setzt taegliche Limits zurueck (nicht ausgefuhrte Orders, etc.)."""
        self._symbol_traded_today.clear()
        self._last_exit_time.clear()  # Cooldowns verfallen ueber Nacht

    def get_status(self) -> dict:
        """Gibt aktuellen Status fuer Logging zurueck."""
        return {
            "open_positions": self.open_positions,
            "symbols": self.open_symbols,
            "max_positions": self.max_positions,
            "traded_today": len(self._symbol_traded_today),
        }
