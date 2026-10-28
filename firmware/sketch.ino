#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define PIN_R   26
#define PIN_G   27
#define PIN_B   14
#define PIN_BZ  25
#define PIN_BTN 4

Adafruit_SSD1306 oled(128, 64, &Wire, -1);
const uint8_t MPU_ADDR = 0x68;

const int   N_SAMPLES   = 25;
const int   CALIB_WIN   = 6;
const float WARN_RATIO  = 2.5;
const float CRIT_RATIO  = 6.0;

enum Mode { MODE_NORMAL, MODE_INSTABLE, MODE_PANNE };
const char* modeNames[]  = { "NORMALE", "INSTABLE", "PANNE" };
const float modeAmp[]    = { 0.02, 0.08, 0.35 };
Mode mode = MODE_NORMAL;

enum Level { LVL_OK, LVL_WARN, LVL_CRIT };
const char* levelNames[] = { "NORMAL", "ATTENTION", "CRITIQUE" };

float buf[N_SAMPLES];
int   idx = 0;
float baseline = 0;
int   calibCount = 0;
bool  calibrated = false;
Level level = LVL_OK;

struct Alert { uint32_t t; Level lvl; float ratio; };
Alert history[4];
int   histCount = 0;

float lastStd = 0, lastP2P = 0, lastRatio = 0;

void mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0);
  Wire.endTransmission(true);
}

float readAccelMagnitude() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 6, (int)true);
  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();
  float gx = ax / 16384.0, gy = ay / 16384.0, gz = az / 16384.0;
  return sqrt(gx * gx + gy * gy + gz * gz);
}

float injectVibration(float t) {
  float amp = modeAmp[mode];
  float noise = (random(-1000, 1000) / 1000.0) * amp;
  float harmonic = 0.5 * amp * sin(2 * PI * 35 * t);
  float burst = 0;
  if (mode == MODE_PANNE && random(0, 100) < 8) burst = amp * 2.0;
  return noise + harmonic + burst;
}

void computeStats() {
  float mean = 0, mn = buf[0], mx = buf[0];
  for (int i = 0; i < N_SAMPLES; i++) {
    mean += buf[i];
    if (buf[i] < mn) mn = buf[i];
    if (buf[i] > mx) mx = buf[i];
  }
  mean /= N_SAMPLES;
  float var = 0;
  for (int i = 0; i < N_SAMPLES; i++) var += (buf[i] - mean) * (buf[i] - mean);
  lastStd = sqrt(var / N_SAMPLES);
  lastP2P = mx - mn;
}

void analyze() {
  computeStats();

  if (!calibrated) {
    baseline += lastStd;
    if (++calibCount >= CALIB_WIN) {
      baseline /= CALIB_WIN;
      if (baseline < 0.005) baseline = 0.005;
      calibrated = true;
    }
    return;
  }

  lastRatio = lastStd / baseline;
  Level newLevel = LVL_OK;
  if (lastRatio >= CRIT_RATIO)      newLevel = LVL_CRIT;
  else if (lastRatio >= WARN_RATIO) newLevel = LVL_WARN;

  if (newLevel != level) {
    level = newLevel;
    for (int i = 3; i > 0; i--) history[i] = history[i - 1];
    history[0] = { millis() / 1000, level, lastRatio };
    if (histCount < 4) histCount++;
  }

  Serial.printf("%lu,%s,%.4f,%.4f,%.2f,%s\n", millis(), modeNames[mode],
                lastStd, lastP2P, lastRatio, levelNames[level]);
}

void updateLED() {
  digitalWrite(PIN_R, level == LVL_WARN || level == LVL_CRIT);
  digitalWrite(PIN_G, level == LVL_OK   || level == LVL_WARN);
  digitalWrite(PIN_B, LOW);
}

void updateBuzzer() {
  uint32_t t = millis();
  if (level == LVL_OK) {
    noTone(PIN_BZ);
  } else if (level == LVL_WARN) {
    if ((t / 1000) % 2 == 0 && (t % 1000) < 100) tone(PIN_BZ, 1200);
    else noTone(PIN_BZ);
  } else {
    if ((t / 200) % 2 == 0) tone(PIN_BZ, 2400);
    else noTone(PIN_BZ);
  }
}

void drawScreen() {
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0);
  oled.printf("SentinelEdge [%s]", modeNames[mode]);
  oled.drawLine(0, 9, 127, 9, SSD1306_WHITE);

  if (!calibrated) {
    oled.setCursor(0, 20); oled.print("Calibration...");
    oled.setCursor(0, 32); oled.printf("%d/%d", calibCount, CALIB_WIN);
  } else {
    oled.setCursor(0, 12); oled.printf("Etat: %s", levelNames[level]);
    oled.setCursor(0, 22); oled.printf("Vib: %.3fg  x%.1f", lastStd, lastRatio);
    oled.setCursor(0, 32); oled.printf("P2P: %.3fg", lastP2P);
    oled.setCursor(0, 42); oled.print("Historique:");
    for (int i = 0; i < min(histCount, 2); i++) {
      oled.setCursor(0, 51 + i * 8);
      oled.printf("%lus %s x%.1f", history[i].t, levelNames[history[i].lvl], history[i].ratio);
    }
  }
  oled.display();
}

void checkButton() {
  static bool last = HIGH;
  static uint32_t lastT = 0;
  bool now = digitalRead(PIN_BTN);
  if (last == HIGH && now == LOW && millis() - lastT > 250) {
    mode = (Mode)((mode + 1) % 3);
    lastT = millis();
  }
  last = now;
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_R, OUTPUT); pinMode(PIN_G, OUTPUT); pinMode(PIN_B, OUTPUT);
  pinMode(PIN_BZ, OUTPUT);
  pinMode(PIN_BTN, INPUT_PULLUP);
  Wire.begin(21, 22);
  mpuInit();
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  Serial.println("ms,mode,std_g,p2p_g,ratio,niveau");
}

void loop() {
  static uint32_t lastSample = 0;
  checkButton();

  if (millis() - lastSample >= 20) {
    lastSample = millis();
    buf[idx++] = readAccelMagnitude() + injectVibration(millis() / 1000.0);
    if (idx >= N_SAMPLES) {
      idx = 0;
      analyze();
      updateLED();
      drawScreen();
    }
  }
  if (calibrated) updateBuzzer();
}
