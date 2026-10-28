# SentinelEdge

Maintenance predictive sur ESP32 : analyse de vibrations, FFT, detection d'anomalies (seuil adaptatif et Isolation Forest), estimation du temps avant panne (RUL) et tableau de bord temps reel.

Le projet simule une machine industrielle dans [Wokwi](https://wokwi.com), la surveille avec un accelerometre, et detecte les comportements anormaux avant la panne.

## Sommaire

1. Fonctionnalites
2. Architecture
3. Materiel simule
4. Structure du depot
5. Demarrage rapide
6. Format des donnees
7. Methodes utilisees
8. Tableau de bord
9. API
10. Limites
11. Pistes d'amelioration

## 1. Fonctionnalites

**Firmware (ESP32)**
- Lecture du MPU6050 en I2C (registres bruts) a 200 Hz.
- Calcul de l'ecart-type et du pic-a-pic sur une fenetre de 64 echantillons.
- Calibration automatique : l'ESP32 apprend la vibration normale au demarrage.
- Analyse frequentielle (DFT) pour extraire la frequence dominante.
- Trois niveaux de gravite (normal, attention, critique) avec LED RGB et buzzer.
- Historique des alertes sur l'ecran OLED.
- Quatre modes de machine simulee : normale, instable, panne, derive progressive.
- Export CSV sur le port serie et envoi WiFi optionnel vers l'API.

**Data science (Python)**
- Detection d'anomalies par Isolation Forest, entraine sur la machine saine.
- Score de sante de 0 a 100.
- Estimation du temps avant panne (RUL) par tendance lineaire.
- Comparaison entre le seuil embarque et l'IA (avance de detection).
- Tableau de bord interactif avec lecture en direct.

## 2. Architecture

```
  MPU6050 --I2C--> ESP32 ----> OLED / LED RGB / Buzzer
                     |
                     |  CSV serie ou HTTP (JSON)
                     v
              API FastAPI ---> SQLite
                     |
                     v
         Dashboard Streamlit
   (Isolation Forest, sante, RUL, spectrogramme)
```

## 3. Materiel simule

| Composant | Role | Broches ESP32 |
|---|---|---|
| ESP32 DevKit C | Microcontroleur | - |
| MPU6050 | Accelerometre | SDA 21, SCL 22 |
| Ecran OLED SSD1306 | Affichage de l'etat | SDA 21, SCL 22 |
| LED RGB | Indicateur de gravite | 26 (R), 27 (V), 14 (B) |
| Buzzer | Alarme sonore | 25 |
| Bouton | Changement de mode | 4 |

## 4. Structure du depot

```
sentinel-edge-predictive-maintenance/
├── firmware/
│   ├── sketch.ino
│   ├── diagram.json
│   └── libraries.txt
├── backend/
│   ├── api.py
│   └── import_csv.py
├── dashboard/
│   └── dashboard.py
├── data/
│   └── mesures.csv
├── docs/
│   └── (captures d'ecran)
├── requirements.txt
└── README.md
```

## 5. Demarrage rapide

### Firmware sur Wokwi

1. Cree un projet ESP32 sur wokwi.com.
2. Remplace `sketch.ino`, `diagram.json` et `libraries.txt` par ceux du dossier `firmware/`.
3. Lance la simulation. Apres environ 3 secondes de calibration, l'OLED affiche l'etat.
4. Appuie sur le bouton pour changer de mode :

| Mode | Vibration | Frequence | Resultat attendu |
|---|---|---|---|
| Normale | faible | 25 Hz | LED verte, silence |
| Instable | moyenne | 50 Hz | LED orange, bip court |
| Panne | forte, avec chocs | 78 Hz | LED rouge, alarme continue |
| Derive | croissante sur 60 s | 25 a 78 Hz | passage progressif de vert a rouge |

5. Copie la sortie du moniteur serie dans un fichier `mesures.csv`.

### Python

```bash
pip install -r requirements.txt
python -m streamlit run dashboard/dashboard.py
```

Dans la barre laterale, choisis la source **Demonstration** pour voir le tableau de bord sans Wokwi, ou **Fichier CSV** pour charger ton `mesures.csv`.

### API (optionnelle)

```bash
python -m uvicorn api:app --port 8000
python import_csv.py mesures.csv
```

Puis choisis la source **API (SQLite)** dans le tableau de bord. La documentation interactive est disponible sur `http://localhost:8000/docs`.

### Envoi WiFi en temps reel (optionnel)

Wokwi ne peut pas joindre `localhost`. Expose l'API avec un tunnel (`ngrok http 8000`), copie l'URL HTTPS dans `SERVER_URL` du sketch, puis passe `USE_WIFI` a `true`.

## 6. Format des donnees

Le firmware ecrit une ligne CSV par fenetre de mesure :

```
ms,mode,std_g,p2p_g,ratio,freq_hz,niveau
5266,NORMALE,0.0119,0.0527,0.96,25.0,NORMAL
```

| Colonne | Signification |
|---|---|
| `ms` | Temps ecoule depuis le demarrage (millisecondes) |
| `mode` | Mode de la machine simulee |
| `std_g` | Ecart-type de la vibration (g) |
| `p2p_g` | Amplitude pic-a-pic (g) |
| `ratio` | `std_g` divise par la vibration normale apprise |
| `freq_hz` | Frequence dominante (Hz) |
| `niveau` | NORMAL, ATTENTION ou CRITIQUE |

Le tableau de bord accepte aussi l'ancien format a 6 colonnes (sans `freq_hz`).

## 7. Methodes utilisees

**Seuil adaptatif.** Les six premieres fenetres servent a calculer une vibration de reference. Le ratio vibration / reference est compare a deux seuils : 2,5 pour ATTENTION et 6 pour CRITIQUE. Le seuil s'adapte donc a chaque machine.

**Frequence dominante.** Une DFT sur 64 points a 200 Hz donne une resolution d'environ 3,1 Hz. La frequence dominante identifie le type de defaut, ce que l'intensite seule ne permet pas.

**Isolation Forest.** Le modele est entraine sur les mesures de la machine normale (`std_g`, `p2p_g`, `freq_hz`). Un score bas signale une mesure inhabituelle, sans seuil fixe a regler.

**Score de sante.** Combinaison de 70 % du ratio de vibration et de 30 % du score Isolation Forest, ramenee entre 0 et 100, puis lissee par moyenne mobile exponentielle.

**Temps avant panne (RUL).** Regression lineaire du score de sante sur une fenetre glissante, extrapolee jusqu'au seuil de panne choisi.

## 8. Tableau de bord

- Bandeau d'etat colore, qui pulse en cas d'etat critique.
- Indicateurs : sante, RUL, vibration, frequence, alertes.
- Jauges de sante et de ratio de vibration.
- Courbe de sante avec projection de panne et marqueurs de changement d'etat.
- Spectrogramme montrant la derive de la frequence du defaut.
- Nuage 3D des caracteristiques colore par le score de l'IA.
- Comparaison seuil embarque et IA : premiere detection de chacun et avance obtenue.
- Lecture en direct : rejoue les mesures en temps reel, avec reglage de la vitesse, pause et relecture.

## 9. API

| Methode | Route | Description |
|---|---|---|
| POST | `/measure` | Enregistre une mesure |
| POST | `/measures/batch` | Enregistre une liste de mesures |
| GET | `/measures?limit=2000` | Derniere mesures |
| GET | `/summary` | Statistiques par mode |
| DELETE | `/measures` | Vide la base |

## 10. Limites

- Les vibrations sont synthetiques : le capteur de Wokwi ne mesure pas une vraie machine.
- Le temps affiche est celui de la simulation, qui peut etre plus lent que le temps reel.
- Le RUL est une extrapolation lineaire. Sur une degradation acceleree, il est optimiste.
- La DFT est calculee directement. Une FFT optimisee serait preferable sur un vrai capteur.

## 11. Pistes d'amelioration

- Remplacer le RUL lineaire par un modele non lineaire ou un filtre de Kalman.
- Transmettre les mesures par MQTT.
- Utiliser FreeRTOS pour separer acquisition, analyse et communication.
- Tester sur un vrai ESP32 avec un MPU6050 fixe sur un moteur.
- Ajouter des alertes par e-mail ou message.

## Licence

MIT
