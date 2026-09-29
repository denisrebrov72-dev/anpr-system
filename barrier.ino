#include <Servo.h>

// ─── Пины ───
const int SERVO_PIN = 9;        // сигнальный провод сервы (оранжевый)
const int TRIG_PIN  = 7;        // HC-SR04 TRIG
const int ECHO_PIN  = 8;        // HC-SR04 ECHO

// ─── Параметры сервы ───
const int ANGLE_CLOSED = 0;     // угол «закрыто»
const int ANGLE_OPEN   = 90;    // угол «открыто»
const unsigned long OPEN_DURATION_MS = 10000;  // 10 сек держать открытым

// ─── Объекты ───
Servo barrierServo;
unsigned long open_started_at = 0;
bool is_open = false;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);

  barrierServo.attach(SERVO_PIN);
  barrierServo.write(ANGLE_CLOSED);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  Serial.println("READY");
}

void loop() {
  // Чтение команд из Serial
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd == "OPEN") {
      open_barrier();
      Serial.println("OK:OPEN");
    } else if (cmd == "CLOSE") {
      close_barrier();
      Serial.println("OK:CLOSE");
    } else if (cmd == "PING") {
      Serial.println("PONG");
    } else if (cmd == "DIST") {
      long d = measure_distance_cm();
      Serial.print("DIST:");
      Serial.println(d);
    } else if (cmd.length() > 0) {
      Serial.print("ERR:UNKNOWN:");
      Serial.println(cmd);
    }
  }

  // Автозакрытие через OPEN_DURATION_MS
  if (is_open && (millis() - open_started_at >= OPEN_DURATION_MS)) {
    close_barrier();
    Serial.println("AUTO:CLOSE");
  }
}

void open_barrier() {
  barrierServo.write(ANGLE_OPEN);
  open_started_at = millis();
  is_open = true;
}

void close_barrier() {
  barrierServo.write(ANGLE_CLOSED);
  is_open = false;
}

long measure_distance_cm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);  // таймаут 30 мс (~5 м)
  if (duration == 0) return -1;                     // нет отклика

  long distance = duration * 0.034 / 2;
  return distance;
}