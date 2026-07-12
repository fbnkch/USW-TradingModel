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
    breakeven_triggered: bool = False
    breakeven_price: float = 0.0
    vwap_at_entry: float = 0.0    # Session-VWAP zum Entry-Zeitpunkt
    time_stop_minutes: int = 30
    order_id: str = ""
    current_finder_score: float = 0.0


@dataclass
class ExitAction:
    """Beschreibt eine auszufuehrende Exit-Order."""
    symbol: str
    reason: str  # "take_profit", "ratchet_exit", "trailing_stop", "stop_loss", "time_stop", "signal_collapse", "partial_profit"
    entry_price: float
    current_price: float
    pnl_pct: float
    partial_qty: int = 0  # > 0 bei Teilgewinnmitnahme: nur diese Stueckzahl verkaufen


class AccountManager:
    """Verwaltet Positionen und Risk-Management."""

    def __init__(
        self,
        trading_client,
        max_positions: int = 10,
        max_risk_per_trade: float = 0.005,
        tp_pct: float = 0.005,
        sl_pct: float = 0.0060,
        time_stop_minutes: int = 30,
        signal_collapse_threshold: float = 0.20,
        position_size_pct: float = 0.05,
        enable_trailing_sl: bool = True,
        reentry_cooldown_minutes: int = 5,
        sl_time_decay_target: float = 0.003,
        sl_time_decay_grace: int = 5,
        grace_sl_pct: float = 0.015,
        grace_sl_pct_t2: float = 0.010,
        entry_grace_minutes: int = 3,
        entry_grace_t2_minutes: int = 5,
        atr_trail_start_mult: float = 2.5,
        atr_trail_min_mult: float = 1.2,
        atr_trail_ramp_pct: float = 0.015,
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
        self.enable_trailing_sl = enable_trailing_sl
        self.reentry_cooldown = timedelta(minutes=reentry_cooldown_minutes)
        self.sl_time_decay_target = sl_time_decay_target
        self.sl_time_decay_grace = sl_time_decay_grace
        self.grace_sl_pct = grace_sl_pct
        self.grace_sl_pct_t2 = grace_sl_pct_t2
        self.entry_grace_minutes = entry_grace_minutes
        self.entry_grace_t2_minutes = entry_grace_t2_minutes
        # ATR-basierter Trailing Stop (Zarattini et al. 2024)
        self.atr_trail_start_mult = atr_trail_start_mult
        self.atr_trail_min_mult = atr_trail_min_mult
        self.atr_trail_ramp_pct = atr_trail_ramp_pct
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

    def _current_trail_distance_pct(self, pos: Position, atr: float) -> float:
        """ATR-basierter Trailing-Stop-Abstand (Zarattini et al. 2024).

        Research: Statt fixer Prozentwerte wird der Trail-Abstand aus der
        aktuellen Volatilitaet (ATR) abgeleitet — passt sich automatisch
        an jedes Symbol an.

        Je weiter der Kurs ueber Entry gestiegen ist, desto enger der Trail:
          - Bei Entry (Profit=0):     Trail = atr_trail_start_mult * ATR
          - Bei voller Rampe:         Trail = atr_trail_min_mult * ATR

        Bsp. MU  (ATR~1.5%): Trail startet bei 3.75%, minimum 1.8%
        Bsp. LCID (ATR~0.3%): Trail startet bei 0.75%, minimum 0.36%

        Der Trail-Abstand schrumpft linear mit dem highest_price.
        """
        if atr <= 0:
            return 0.008

        profit_at_peak = (pos.highest_price - pos.entry_price) / pos.entry_price
        if profit_at_peak <= 0:
            return self.atr_trail_start_mult * atr

        ramp = min(1.0, profit_at_peak / self.atr_trail_ramp_pct)
        mult = self.atr_trail_start_mult + (self.atr_trail_min_mult - self.atr_trail_start_mult) * ramp
        return max(self.atr_trail_min_mult * atr, mult * atr)

    def check_exits(
        self,
        current_prices: dict[str, float],
        atr_values: dict[str, float] | None = None,
        vwap_values: dict[str, float] | None = None,
    ) -> list[ExitAction]:
        """Prueft alle offenen Positionen auf Exit-Signale.

        STATE-OF-THE-ART EXIT-SYSTEM (Zarattini, Aziz & Barbon 2024):

        1. Grace-SL (0-5 Min):        1.5% -> 1.0% Crash-Schutz
        2. ATR-Trailing-Stop:          Dynamischer Stop basierend auf 14er-ATR.
           Startet bei 2.5xATR, zieht sich auf 1.2xATR zusammen.
           KEINE fixen TP-Level - der ATR-Stop laesst Gewinner laufen.
        3. VWAP-Floor:                 Session-VWAP als zusaetzlicher Boden.
           "Exit when price falls below the higher of Band or VWAP" (Paper)
        4. Time-Decay SL:              Zieht sich linear mit der Zeit zusammen
        5. Breakeven-Stop (konservativ): Nur nach 10+ Min, nur bei +0.4%

        KEIN fixer Ratchet/TP - der ATR-Trailing-Stop uebernimmt die
        Gewinnsicherung dynamisch, ohne Gewinner bei +0.5% zu deckeln.

        Effektiver SL = max(alle Schichten). Kann NUR steigen, NIE fallen.
        """
        now = datetime.now(timezone.utc)
        exits = []

        if atr_values is None:
            atr_values = {}
        if vwap_values is None:
            vwap_values = {}

        BREAKEVEN_THRESHOLD = 0.004   # +0.40%
        BREAKEVEN_MIN_AGE = 10        # Minuten

        for sym, pos in list(self._positions.items()):
            if sym not in current_prices:
                continue

            price = current_prices[sym]
            age = now - pos.entry_time
            pnl = (price - pos.entry_price) / pos.entry_price
            age_minutes = age.total_seconds() / 60.0
            atr = atr_values.get(sym, 0.008)

            # --- Highest-Price-Tracking (fuer ATR-Trail) ---
            if price > pos.highest_price:
                pos.highest_price = price

            current_vwap = vwap_values.get(sym, 0.0)

            # --- Breakeven-Stop (konservativ, nur nach 10+ Min etabliert) ---
            if (not pos.breakeven_triggered
                    and age_minutes >= BREAKEVEN_MIN_AGE
                    and price >= pos.entry_price * (1.0 + BREAKEVEN_THRESHOLD)):
                pos.breakeven_triggered = True
                pos.breakeven_price = pos.entry_price
                print(f"  [BREAKEVEN] {sym}: +0.40% nach {age_minutes:.0f}m! "
                      f"SL auf Entry. Trade etabliert, kein Verlust mehr.")

            # --- Grace-SL (Crash-Schutz in den ersten Minuten) ---
            in_grace = age_minutes < self.entry_grace_t2_minutes
            if age_minutes < self.entry_grace_minutes:
                active_sl_pct = self.grace_sl_pct
            elif age_minutes < self.entry_grace_t2_minutes:
                active_sl_pct = self.grace_sl_pct_t2
            else:
                active_sl_pct = self.sl_pct

            # --- Time-Decay SL ---
            pos_time_stop = float(getattr(pos, 'time_stop_minutes', self.time_stop_minutes))
            if in_grace:
                time_sl = 0.0
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
            be_price = pos.breakeven_price if pos.breakeven_triggered else 0.0

            # --- ATR-Trailing-Stop (Kerninnovation) ---
            atr_trail_distance = self._current_trail_distance_pct(pos, atr)
            atr_trailing_sl = pos.highest_price * (1.0 - atr_trail_distance)

            # VWAP-Floor (Paper: "higher of Upper Band or VWAP")
            vwap_floor = 0.0
            if current_vwap > 0 and pos.highest_price > pos.entry_price:
                vwap_floor = max(pos.entry_price, current_vwap)

            # Effektiver SL — KEIN Ratchet-Floor, der ATR-Trail uebernimmt alles
            effective_sl = max(
                base_sl,
                be_price,
                time_sl,
                atr_trailing_sl,
                vwap_floor,
            )

            # --- EXIT-CHECKS ---

            # ATR-Trailing-Stop: Gewinner dynamisch schuetzen
            if (self.enable_trailing_sl
                    and pos.highest_price > pos.entry_price
                    and price <= effective_sl
                    and effective_sl > pos.entry_price):
                exits.append(ExitAction(sym, "trailing_stop", pos.entry_price, price, pnl))

            # Stop Loss (fixer SL / Grace / Time-Decay / VWAP)
            elif price <= effective_sl:
                exits.append(ExitAction(sym, "stop_loss", pos.entry_price, price, pnl))

            # Time Stop
            elif age >= timedelta(minutes=pos_time_stop):
                exits.append(ExitAction(sym, "time_stop", pos.entry_price, price, pnl))

            # Signal Collapse
            elif pos.current_finder_score < self.signal_collapse_threshold and age > timedelta(minutes=5):
                exits.append(ExitAction(sym, "signal_collapse", pos.entry_price, price, pnl))

        return exits

    def register_exit(self, action: ExitAction):
        """Registriert einen Exit.

        Bei Teilgewinnmitnahme (partial_qty > 0): Position wird nicht geschlossen,
        sondern nur die Quantity reduziert. Der Rest laeuft mit Ratchet weiter.
        """
        if action.symbol not in self._positions:
            return

        if action.partial_qty > 0:
            # Teilgewinnmitnahme: Position verkleinern, nicht schliessen
            pos = self._positions[action.symbol]
            pos.qty -= action.partial_qty

            if self.logger:
                partial_pnl = (action.current_price - pos.entry_price) * action.partial_qty
                print(f"  [PARTIAL FILL] {action.symbol}: "
                      f"{action.partial_qty} Shares verkauft @ ${action.current_price:.2f} "
                      f"(PnL=${partial_pnl:+.2f}), Rest: {pos.qty} Shares "
                      f"mit SL=${pos.ratchet_floor:.2f}")

            # Wash-Trade-Schutz: Exit-Zeitpunkt merken
            self._last_exit_time[action.symbol] = datetime.now(timezone.utc)
            return

        # Voller Exit: Position entfernen
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
