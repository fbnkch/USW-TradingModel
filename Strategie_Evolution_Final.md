# Strategie-Evolution

---

Unser Trading-System durchlief quasi "drei" Entwicklungsstufen. Anfangs hatten wir eine einfache
Breakout-Strategie mit fixen Stop-Losses, danach entwickelten wir durch Analysen von unseren Tagesergebnissen und mithilfe von aktuellen Research-Papern
ein neues System, das ebenfalls mit neueren Tagesergebnissen immer weiter verbessert/angepasst wurde.

```
V1 (Baseline):       V2 (ATR-Trail):        V3 (stetige Optimierung):
Fixe Stops            ATR-Trailing-Stop      V2 + Parameter-Tuning
TP 0.5% / SL 0.6%     Zarattini et al.       SL-Weitung, Time-Stop,
Keine Filter           4 Entry-Filter         Open-30-Filter
Ratchet-System         Quality-Floors         MLP-Threshold 0.92
```

---

## Details

### #1: Baseline 30.6.- 3.7.

**Ansatz:** Klassische Breakout-Strategie mit Ensemble aus 5 Modellen (MLP-Gate +
LSTM, GRU, CNN, LightGBM). Fixe Take-Profit- und Stop-Loss-Level für alle Symbole.
Einfaches Ratchet-System zur Gewinnsicherung.

**Schwächen/Beobachtungen:**
- Fixe Prozent-Stops ignorieren die Volatilitätsunterschiede zwischen Symbolen
  (LCID mit 0,22% ATR bekommt denselben Stop wie GFS mit 0,80% ATR)
- Keine Entry-Filter -> viele Fehlsignale in volatilen Phasen
- 66% aller Exits sind Stop-Losses
- Take-Profit bei +0,5% deckelt Gewinner künstlich

**Ergebnisse:**

| Metrik | Wert                          |
|---|-------------------------------|
| Trades | 326                           |
| Win-Rate | ~31%                          |
| Total P&L | **−$2.452 (−24,5% auf $10k)** |
| Profit-Faktor | 0,31                          |

---

### V2: ATR-Trailing-Stop + Entry-Filter (ab 4.7.)

basierend auf Zarattini, Aziz & Barbon (2024): _"Beat the
Market: An Effective Intraday Momentum Strategy for S&P500 ETF"_

**zentrale Änderungen:**

1. **ATR-Trailing-Stop**: Dynamischer Stop-Abstand pro Symbol. Bei Entry 2,5× ATR
   (Atemluft), ab +1,5% nur noch 1,2× ATR (enge Gewinnsicherung). Linear interpoliert.
   Beispiel: LCID (ATR 0,22%): Trail = 0,55% --> 0,26%. GFS (ATR 0,80%): Trail = 2,00% --> 0,96%.

2. **VWAP-Floor**: Stop rutscht nie unter Session-VWAP. Verhindert verfrühte Exits
   in laufenden Trends.

3. **Vier Entry-Filter:**
   - **QQQ-Marktkontext:** Keine Longs bei fallendem Nasdaq-100 (hier vielleicht zeigen wie sich Nasdaq allgemein verhalten hat während unseres Zeitraums)
   - **Volumen-Bestätigung:** `volume_norm > 0` (Breakout mit Volumen)
   - **Finder-Agreement 3/4:** 3 von 4 Findern müssen zustimmen (vorher 2/4)
   - **Streak-Filter:** 3-Minuten-Sperre pro Symbol nach Signal

4. **Quality-Floors pro Marktregime:**
   - Open Drive (9:30–10:30 ET): QF 0,40
   - Late Morning (10:30–12:00): QF 0,50
   - Midday (12:00–14:00): QF 0,48 + 40% Positionsgröße
   - Afternoon (14:00–15:00): QF 0,40

**Ergebnis (V1 vs erste Version von der advanced Version):**

| Metrik | V1 (Baseline) | V2 (ATR-Trail am 4.7.) | Verbesserung      |
|---|--------------|------------------------|-------------------|
| Trades | 326          | 27                     | —                 |
| Win-Rate | 31%          | **44%**                | **+13%**          |
| Total P&L | −$2.452      | −$443                  | **+$2.009 (82%)** |
| Trailing-Stop-WR | —            | **92,3%**              | —                 |
| Profit-Faktor | 0,31         | 0,22                   | —                 |

**Analyse:** Die Win-Rate stieg stark an (+13%). Der ATR-Trailing-Stop
funktionierte hervorragend: 12 von 13 Trailing-Stop-Exits waren Gewinner
(92,3% WR). Allerdings blieb das Risk/Reward-Verhältnis trotz ATR-Trailing-Stop ungünstig: die wenigen Stop-Losses (−$40 avg) fraßen die vielen kleinen Gewinne
(+$9 avg) vollständig auf --> Verhältnis 1:0.23. Zudem fehlte eine Time-Stop-Komponente für Trend-Trades.
Profit Faktor schlechter als V1 aber weniger Verlust --> weil weitaus weniger Trades 

**Lessons Learned vom ersten Run von V2:**
1. weiten von initial SL --> zeit geben damit ATR-Trail greifen kann
2. Time-Stop für Trendfolge-Trades reaktivieren
3. Erste 30 Minuten nach Market Open besonders verlustreich (14 Trades, −$312) --> eigentlich immer gute Chancen auf Gewinne
4. MLP-Gate bei 0,85 quasi wirkungslos (Avg = 0,96 --> 100% der Signale positiv)

---

### V3: Optimierte Parameter 

**Vier Parameter-Änderungen aus der V2-Analyse abgeleitet:**

1. **Stop-Loss: von −0,6%** --> Grace-SL (1,5% bis 1,0% bis 0,6% je nach dem wie lange wir halten). Theoretisch würde damit TP/SL Verhältnis weiter sinken, wir haben aber durch ATR-Trailing-Stop keinen fixen TP mehr. Also: dynamisch Gewinne >0,5% einsammeln + weniger zu schnelle Dropouts (von 52% --> 22%) dadurch mehr Trades mit gutem Trail.

2. **Time-Stop (30 Min) aktiviert**: für Trades die 10+ Minuten überleben
   und >0,3% im Plus sind. Der ATR-Trail sichert weiterhin ab, aber der Time-Stop
   erlaubt Trend-Trades, voll zu laufen.

3. **Open-30-Filter**: Quality-Floor 0,55 in den ersten 30 Minuten nach Market
   Open (15:30–16:00 MESZ). Filtert ~15% der schlechtesten Early-Trades heraus,
   die für −$200 der −$312 Early-Verluste verantwortlich waren.

4. **MLP-Threshold: 0,85 → 0,92**: Bei einem MLP-Mean von 0,96 filtert der
   höhere Threshold ~10% der schwächsten Signale heraus. Verbessert die
   Signalqualität ohne wirklichen Trade-Verlust.

**Ergebnis:**

| Metrik            | V1 (Baseline) | V2 (ATR-Trail am 4.7.) | V3 (Optimierungen ab 5.7.) |
|-------------------|---------------|------------------------|----------------------------|
| Trades            | 326           | 27                     | 115 auf 5 Tage             |
| Win-Rate          | ~31%          | 44,4%                  | **~60-65%**                |
| Total P&L         | −$2.452       | −$443                  | **+$1364**                 |
| avg Rendite (Tag) | −2,45%        | −0,44%                 | **+1,2%**                  |
| Profit-Faktor     | 0,31          | 0,22                   | **1,3 – 1,5**              |
| Stop-Loss-Rate    | ~66%          | 52%                    | **~22%**                   |

---

## Exit-Gründe im Vergleich

| Exit-Reason | V1 | V2 (4.7.)             | V3 (ab 5.7.)                |
|---|---|-----------------------|-----------------------------|
| Stop-Loss | ~66% (−$14 avg) | 52% (−$40 avg)        | ~22% (−$40-45 avg)          |
| Trailing-Stop | — | 48% (+$9 avg, 92% WR) | ~57% (+$5-10 avg, ~93% WR)  |
| Time-Stop | — | —                     | ~13% (+$40–50 avg, 100% WR) |
| Ratchet | ~34% (+$5 avg) | —                     | ~8% (+$5-10 avg)            |

**Erkenntnis:** Die Kombination aus ATR-Trail und
Time-Stop liefert das beste Risiko/Rendite-Profil.


## Was wir gelernt haben

1. **ATR-Trailing-Stops sind den fixen Stops überlegen.** Die 92,3% Win-Rate auf
   Trailing-Stop-Exits und die 82%ige Verlustreduktion zeigen das
   eindeutig. Allerdings muss das Gewinn/Verlust-Verhältnis pro Trade stimmen.

2. **Entry-Filter verbessern die Signalqualität messbar.** Die vier neuen Filter
   (QQQ, Volumen, 3/4 Votes, Streak) reduzierten die Trade-Anzahl stark, sorgen aber gleichzeitig für höhere Win-Rate.

3. **Hybride Exit-Strategie ist besser.** ATR-Trail für die Mehrheit der Trades + Time-Stop für die stärksten Trendfolger kombiniert die Vorteile: enge Verlustkontrolle und Teilnahme an großen Moves.

4. **Iterative Analyse zahlt sich aus.** Jeden Tag Werte gesammelt und am Ende der Perioden vollständig ausgewertet um Anpassungen zu tätigen. 
Am Ende sieht man deutliche Verbesserungen zu erster Strategie --> mehr Zeit hätte noch weitaus bessere Ergebnisse geliefert.


