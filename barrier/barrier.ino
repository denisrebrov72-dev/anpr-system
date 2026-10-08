// ════════════════════════════════════════════════════════════════
//  ANPR — индикация доступа (зелёный / красный светодиод)
//  Протокол: PING → PONG, GREEN → OK:GREEN, RED → OK:RED
// ════════════════════════════════════════════════════════════════

const int LED_GREEN = 8;   // зелёный — номер свой
const int LED_RED   = 7;   // красный — номер не свой

const unsigned long HOLD_MS = 3000;  // сколько держать светодиод

unsigned long green_started_at = 0;
unsigned long red_started_at   = 0;
bool green_active = false;
bool red_active   = false;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, LOW);

  Serial.println("READY");
}

void loop() {
  // ─── Приём команд ───
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd == "PING") {
      Serial.println("PONG");
    } else if (cmd == "GREEN") {
      digitalWrite(LED_GREEN, HIGH);
      digitalWrite(LED_RED, LOW);
      green_started_at = millis();
      green_active = true;
      red_active = false;
      Serial.println("OK:GREEN");
    } else if (cmd == "RED") {
      digitalWrite(LED_RED, HIGH);
      digitalWrite(LED_GREEN, LOW);
      red_started_at = millis();
      red_active = true;
      green_active = false;
      Serial.println("OK:RED");
    } else if (cmd == "CLOSE" || cmd == "OFF") {
      digitalWrite(LED_GREEN, LOW);
      digitalWrite(LED_RED, LOW);
      green_active = false;
      red_active = false;
      Serial.println("OK:CLOSE");
    } else if (cmd.length() > 0) {
      Serial.print("ERR:UNKNOWN:");
      Serial.println(cmd);
    }
  }

  // ─── Автовыключение ───
  if (green_active && millis() - green_started_at >= HOLD_MS) {
    digitalWrite(LED_GREEN, LOW);
    green_active = false;
    Serial.println("AUTO:OFF");
  }
  if (red_active && millis() - red_started_at >= HOLD_MS) {
    digitalWrite(LED_RED, LOW);
    red_active = false;
    Serial.println("AUTO:OFF");
  }
}