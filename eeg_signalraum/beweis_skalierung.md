# Der StandardScaler ändert die Eigenwerte $\lambda$ aus $Cv = \lambda Rv$ nicht

Schritt 1 + 2 der SDGL-Rekonstruktion, links ohne, rechts mit Scaler.

## Bezeichnungen

- Fenster mit $N$ Zeitpunkten und $d = 27$ Kanälen. Der Messwert zum Zeitpunkt $t$ ist ein Zeilenvektor $y_t = (y_{t,1}, \dots, y_{t,d}) \in \mathbb{R}^{1 \times d}$, $\;t = 1, \dots, N$.
- Mittelwert je Kanal: $\bar y = \frac{1}{N} \sum_{t=1}^{N} y_t$. Standardabweichung je Kanal: $\sigma_1, \dots, \sigma_d > 0$.
- Skalierungsmatrix $S = \operatorname{diag}\left(\frac{1}{\sigma_1}, \dots, \frac{1}{\sigma_d}\right)$. Sie ist diagonal, also $S^\top = S$, und invertierbar mit $S^{-1} = \operatorname{diag}(\sigma_1, \dots, \sigma_d)$.
- Der StandardScaler: $\tilde y_t = (y_t - \bar y)\,S$. Alles mit Tilde ist die skalierte Version.

## Schritt 1a: zweite Differenzen

| ohne Scaler | mit Scaler |
|---|---|
| $d_j = y_{t_j+2} - 2\,y_{t_j+1} + y_{t_j}, \quad j = 1, \dots, m$ | $\begin{aligned} \tilde d_j &= \tilde y_{t_j+2} - 2\,\tilde y_{t_j+1} + \tilde y_{t_j} \\ &= (y_{t_j+2} - \bar y)S - 2(y_{t_j+1} - \bar y)S + (y_{t_j} - \bar y)S \\ &= (y_{t_j+2} - 2\,y_{t_j+1} + y_{t_j})\,S - (1 - 2 + 1)\,\bar y S \\ &= d_j\,S \end{aligned}$ |

**Warum:** Die Koeffizienten der zweiten Differenz summieren sich zu $1 - 2 + 1 = 0$, deshalb fällt der Mittelwert $\bar y$ weg. $S$ wirkt auf jeden Summanden gleich und lässt sich ausklammern.

## Schritt 1b: Rauschmatrix $R$

| ohne Scaler | mit Scaler |
|---|---|
| $\bar d = \frac{1}{m} \sum_{j=1}^{m} d_j$ | $\overline{\tilde d} = \frac{1}{m} \sum_{j=1}^{m} d_j S = \bar d\,S$ |
| $R = \frac{1}{6} \cdot \frac{1}{m-1} \sum_{j=1}^{m} (d_j - \bar d)^\top (d_j - \bar d)$ | $\begin{aligned} \tilde R &= \tfrac{1}{6} \cdot \tfrac{1}{m-1} \textstyle\sum_{j} (d_j S - \bar d S)^\top (d_j S - \bar d S) \\ &= \tfrac{1}{6} \cdot \tfrac{1}{m-1} \textstyle\sum_{j} S^\top (d_j - \bar d)^\top (d_j - \bar d)\,S \\ &= S \left[ \tfrac{1}{6} \cdot \tfrac{1}{m-1} \textstyle\sum_{j} (d_j - \bar d)^\top (d_j - \bar d) \right] S \\ &= S\,R\,S \end{aligned}$ |

**Warum:** $(aS)^\top = S^\top a^\top = S\,a^\top$, weil $S$ symmetrisch ist. $S$ hängt nicht von $j$ ab und lässt sich links und rechts aus der Summe herausziehen. Der Faktor $\frac{1}{6}$ ist eine Zahl und ändert daran nichts.

## Schritt 2a: Zentrieren und Datenkovarianz $C$

| ohne Scaler | mit Scaler |
|---|---|
| $x_t = y_t - \bar y$ | $\tilde x_t = \tilde y_t - \underbrace{\overline{\tilde y}}_{=0} = (y_t - \bar y)\,S = x_t\,S$ |
| $C = \frac{1}{N-1} \sum_{t=1}^{N} x_t^\top x_t$ | $\begin{aligned} \tilde C &= \tfrac{1}{N-1} \textstyle\sum_{t} (x_t S)^\top (x_t S) \\ &= S \left[ \tfrac{1}{N-1} \textstyle\sum_{t} x_t^\top x_t \right] S = S\,C\,S \end{aligned}$ |

**Warum:** Es ist dieselbe Rechnung wie bei $R$. Zwischenergebnis: Der Scaler wirkt auf **beide** Matrizen gleich, $\tilde C = SCS$ und $\tilde R = SRS$.

## Schritt 2b: verallgemeinertes Eigenwertproblem

| ohne Scaler | mit Scaler |
|---|---|
| $C\,v = \lambda\,R\,v$ | $\tilde C\,w = \tilde\lambda\,\tilde R\,w \;\Longleftrightarrow\; SCS\,w = \tilde\lambda\,SRS\,w$ |
| $q(v) = \dfrac{v^\top C\,v}{v^\top R\,v}$ | $\tilde q(w) = \dfrac{w^\top \tilde C\,w}{w^\top \tilde R\,w} = \dfrac{w^\top SCS\,w}{w^\top SRS\,w} = \dfrac{(Sw)^\top C\,(Sw)}{(Sw)^\top R\,(Sw)} = q(Sw)$ |
| $\lambda_1 = \max_v q(v), \;\; \lambda_2 = \max_{v \perp_R v_1} q(v), \;\dots$ | $\tilde\lambda_1 = \max_w q(Sw), \;\; \tilde\lambda_2 = \dots$ |

**Warum:** $S$ kürzt sich hier nicht wörtlich heraus. Weil Zähler und Nenner mit derselben Matrix $S$ umgeformt werden, lässt sich $S$ in beiden mit $w$ zu $Sw$ zusammenfassen. Der Quotient mit Scaler ist also der alte Quotient, nur an einer anderen Stelle ausgewertet. Weil $S$ invertierbar ist, durchläuft $v = Sw$ alle Richtungen, wenn $w$ alle Richtungen durchläuft. Die Menge der Quotienten ist also dieselbe, und damit auch Maximum und Rangfolge.

## Schritt 2c: Einsetzen, die Eigenwerte sind gleich

| ohne Scaler | mit Scaler |
|---|---|
| $C\,v = \lambda\,R\,v \quad \text{(Annahme)}$ | $\begin{aligned} \text{setze } w &= S^{-1} v: \\ \tilde C\,w &= SCS\,S^{-1} v = S\,C\,v \\ &= S\,\lambda R\,v \\ &= \lambda\,SRS\,S^{-1} v = \lambda\,\tilde R\,w \end{aligned}$ |
| $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$ | $\boxed{\tilde\lambda_i = \lambda_i \quad \text{für alle } i = 1, \dots, d}$ |

**Warum:** Hier kürzt sich $S$ wirklich, an zwei Stellen. Erstens ergibt $S\,S^{-1} = I$, das passiert durch die Wahl $w = S^{-1}v$. Zweitens bleibt $S\,Cv = \lambda\,S\,Rv$ übrig, links steht auf beiden Seiten dasselbe $S$, Multiplikation mit $S^{-1}$ entfernt es. Jede Lösung ohne Scaler liefert so eine Lösung mit Scaler zum selben $\lambda$. Umgekehrt geht es über $v = Sw$.

## Schritt 2d: Richtungen, Normierung und Zeitreihen

| ohne Scaler | mit Scaler |
|---|---|
| $v_i$ (Gewichte über die Kanäle) | $w_i = S^{-1} v_i$ (andere Gewichte) |
| $v^\top R\,v = 1$ | $w^\top \tilde R\,w = v^\top S^{-1}\,SRS\,S^{-1} v = v^\top R\,v = 1$ |
| $z_t = x_t\,v$ | $\tilde z_t = \tilde x_t\,w = x_t S\,S^{-1} v = x_t\,v = z_t$ |

**Warum:** Die Gewichte gleichen die Skalierung genau aus. Die Richtungen sehen anders aus, die projizierten Zeitreihen $Z$ sind identisch.

## Rangprojektion, vor Schritt 2

Vor dem Eigenwertproblem wird $R$ zerlegt und auf eine Basis $P$ der nicht leeren Richtungen projiziert. Dafür gilt allgemein: Für jede invertierbare Matrix $A \in \mathbb{R}^{d \times d}$ haben

$$
(A^\top C A,\; A^\top R A) \quad \text{und} \quad (C,\; R)
$$

dieselben verallgemeinerten Eigenwerte. Der Beweis ist derselbe wie oben, mit $A$ statt $S$. Skalieren mit $S$, eine Drehung $P$ bei vollem Rang und jedes Mischen der Kanäle ändern $\lambda$ also nicht.

Hat $R$ eine fast leere Richtung, etwa durch die Average-Referenz, wird sie entfernt. Das geschieht mit und ohne Scaler. Nur die Entscheidung nahe der Rangschwelle kann sich verschieben.

## Gegenbeispiel: PCA

| ohne Scaler | mit Scaler |
|---|---|
| $C\,v = \mu\,v$ | $\tilde C\,w = \tilde\mu\,w \;\Longleftrightarrow\; SCS\,w = \tilde\mu\,w$ |
| $q_{\mathrm{PCA}}(v) = \dfrac{v^\top C\,v}{v^\top v}$ | $\tilde q_{\mathrm{PCA}}(w) = \dfrac{(Sw)^\top C\,(Sw)}{w^\top w} \;\neq\; q_{\mathrm{PCA}}(Sw)$ |

**Warum:** Im Nenner steht $w^\top w$ statt $w^\top SRS\,w$. Das $S$ im Zähler lässt sich mit $w$ zu $Sw$ zusammenfassen, das im Nenner nicht, weil dort keines steht. Die Eigenwerte von $SCS$ sind im Allgemeinen andere als die von $C$. Gleich bleiben sie nur für $S = c\,I$. Bei PCA ändert der Scaler das Ergebnis, beim verallgemeinerten Problem nicht.

---

**Zuordnung zum Code:**

| Schritt | Datei | Zeilen |
|---|---|---|
| 1a | `stdgl/werkzeuge/diffusionsmatrix.py` | 287 |
| 1b | `stdgl/werkzeuge/diffusionsmatrix.py` | 341–342, `np.cov` geteilt durch 6 |
| 2a–2d | `stdgl/Signalrekonstruktion/funktionen.py` | 257–259, `signalraum` |
| Rangprojektion | `eeg_signalraum/eeg_signalraum.py` | 94–107, `spektrum` |

Numerisch geprüft an einem ABSZ-Anfallsfenster: $\lambda_1 = 27583$ mit und ohne Scaler.
