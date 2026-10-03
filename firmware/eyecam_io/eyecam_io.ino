// EyeCam I/O — one Arduino (Uno/Nano/Pro Micro) for every piece of hardware in the paper:
//
//   * Brain-modulated RGB LED (Fig. 3/4): the LED on the plotter / light wand whose colour
//     the page sets from the live SSVEP strength.            pins 9 (R), 10 (G), 11 (B), PWM
//   * Flickering lamp (Fig. 10/11 real-world rig, Fig. 12 chunks, Fig. 14 world flicker):
//     a high-power LED / lamp through a logic-level MOSFET.  pin 6
//   * LC shutter glasses (Fig. 12/13 SSVEPVMP): both lenses driven in phase so the whole
//     world flickers; each lens connected across two pins and driven with alternating
//     polarity (no DC across the liquid crystal).            lens L: pins 2/3, lens R: pins 4/5
//
// Serial protocol, 115200 baud, one command per line:
//   ?          -> "EYECAM v1"
//   C r g b    set RGB LED (0..255 each)
//   F hz       flicker the lamp at hz (square wave, 50 % duty); F 0 = lamp off
//   L 0|1      lamp steady off/on (stops flicker)
//   S hz       flicker the shutter glasses at hz; S 0 = glasses clear
//   X          everything off
//
// Common-anode RGB LED? set COMMON_ANODE to 1.

#define COMMON_ANODE 0
const uint8_t PIN_R = 9, PIN_G = 10, PIN_B = 11, PIN_LAMP = 6;
const uint8_t LENS_L_A = 2, LENS_L_B = 3, LENS_R_A = 4, LENS_R_B = 5;

float lampHz = 0, shutterHz = 0;
unsigned long lampHalf = 0, shutterHalf = 0, lampT = 0, shutterT = 0;
bool lampOn = false, shutterClosed = false, polarity = false;
char buf[48];
uint8_t blen = 0;

void rgb(int r, int g, int b) {
#if COMMON_ANODE
  r = 255 - r; g = 255 - g; b = 255 - b;
#endif
  analogWrite(PIN_R, r); analogWrite(PIN_G, g); analogWrite(PIN_B, b);
}

void lenses(bool closed) {
  // closed: apply voltage across each lens, flipping polarity every time we close
  if (closed) {
    polarity = !polarity;
    digitalWrite(LENS_L_A, polarity); digitalWrite(LENS_L_B, !polarity);
    digitalWrite(LENS_R_A, polarity); digitalWrite(LENS_R_B, !polarity);
  } else {
    digitalWrite(LENS_L_A, LOW); digitalWrite(LENS_L_B, LOW);
    digitalWrite(LENS_R_A, LOW); digitalWrite(LENS_R_B, LOW);
  }
}

void handle(char *s) {
  char c = s[0];
  if (c == '?') { Serial.println(F("EYECAM v1")); return; }
  if (c == 'C') { int r = 0, g = 0, b = 0; sscanf(s + 1, "%d %d %d", &r, &g, &b); rgb(r, g, b); return; }
  if (c == 'F') {
    lampHz = atof(s + 1);
    if (lampHz > 0) { lampHalf = (unsigned long)(500000.0 / lampHz); lampT = micros(); }
    else { lampOn = false; digitalWrite(PIN_LAMP, LOW); }
    Serial.println(F("ok")); return;
  }
  if (c == 'L') { lampHz = 0; lampOn = atoi(s + 1); digitalWrite(PIN_LAMP, lampOn); Serial.println(F("ok")); return; }
  if (c == 'S') {
    shutterHz = atof(s + 1);
    if (shutterHz > 0) { shutterHalf = (unsigned long)(500000.0 / shutterHz); shutterT = micros(); }
    else { shutterClosed = false; lenses(false); }
    Serial.println(F("ok")); return;
  }
  if (c == 'X') { lampHz = shutterHz = 0; digitalWrite(PIN_LAMP, LOW); lenses(false); rgb(0, 0, 0); Serial.println(F("ok")); return; }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_R, OUTPUT); pinMode(PIN_G, OUTPUT); pinMode(PIN_B, OUTPUT); pinMode(PIN_LAMP, OUTPUT);
  pinMode(LENS_L_A, OUTPUT); pinMode(LENS_L_B, OUTPUT); pinMode(LENS_R_A, OUTPUT); pinMode(LENS_R_B, OUTPUT);
  rgb(40, 0, 0);  // dull red bias glow, like the incandescent bulb of Fig. 2
  lenses(false);
  Serial.println(F("EYECAM v1"));
}

void loop() {
  unsigned long now = micros();
  if (lampHz > 0 && now - lampT >= lampHalf) { lampT += lampHalf; lampOn = !lampOn; digitalWrite(PIN_LAMP, lampOn); }
  if (shutterHz > 0 && now - shutterT >= shutterHalf) { shutterT += shutterHalf; shutterClosed = !shutterClosed; lenses(shutterClosed); }
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') { if (blen) { buf[blen] = 0; handle(buf); blen = 0; } }
    else if (blen < sizeof(buf) - 1) buf[blen++] = ch;
  }
}
