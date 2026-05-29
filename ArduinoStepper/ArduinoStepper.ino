#define PULSE_PIN  9
#define DIR_PIN    8
#define MICROSTEPS 128
#define DIRECTION  LOW
#define STEPS_PER_REV (200 * MICROSTEPS)
#define STEPS_PER_SHOT (STEPS_PER_REV / 45)
#define DEFAULT_RPM 3

float rpm = DEFAULT_RPM;
unsigned long stepPeriodUs = 0;
bool running = false;
String serialBuffer = "";

void recalculate() {
  stepPeriodUs = (unsigned long)(60000000L / (rpm * STEPS_PER_REV));
}

void rotateSteps(long steps) {
  int rampSteps = steps / 3;
  for (long i = 0; i < steps; i++) {
    unsigned long period;
    if (i < rampSteps) {
      period = 1500 + (1500 * 3 * (rampSteps - i) / rampSteps);
    } else if (i > steps - rampSteps) {
      period = 1500 + (1500 * 3 * (i - (steps - rampSteps)) / rampSteps);
    } else {
      period = 1500;
    }
    digitalWrite(PULSE_PIN, HIGH);
    delayMicroseconds(50);
    digitalWrite(PULSE_PIN, LOW);
    delayMicroseconds(period);
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd == "ROTATE") {
    rotateSteps(STEPS_PER_SHOT);
    Serial.println("DONE");
  } else if (cmd == "RUN") {
    running = true;
    Serial.println(">> Running continuously — send STOP to halt");
  } else if (cmd == "STOP") {
    running = false;
    Serial.println(">> Stopped");
  } else if (cmd.startsWith("RPM:")) {
    float val = cmd.substring(4).toFloat();
    if (val > 0) {
      rpm = val;
      recalculate();
      Serial.print(">> RPM set to "); Serial.println(rpm);
    }
  } else if (cmd.startsWith("REPEAT:")) {
    int count = cmd.substring(7).toInt();
    if (count > 0) {
      Serial.print(">> Repeating ");
      Serial.print(count);
      Serial.println(" times");
      for (int i = 0; i < count; i++) {
        rotateSteps(STEPS_PER_SHOT);
        Serial.print(">> Step ");
        Serial.print(i + 1);
        Serial.print(" of ");
        Serial.println(count);
        delay(1000);
      }
      Serial.println("DONE");
    }
  } else if (cmd == "FULL") {
    Serial.println(">> Full rotation, 90 steps");
    for (int i = 0; i < 90; i++) {
      rotateSteps(STEPS_PER_SHOT);
      Serial.print(">> Step ");
      Serial.print(i + 1);
      Serial.println(" of 90");
      delay(1000);
    }
    Serial.println("DONE");
  } else {
    Serial.println("!! Unknown command. Use: ROTATE | FULL | RUN | STOP | RPM:3 | REPEAT:N");
  }
}

void setup() {
  pinMode(PULSE_PIN, OUTPUT);
  pinMode(DIR_PIN,   OUTPUT);
  digitalWrite(DIR_PIN, DIRECTION);
  Serial.begin(9600);
  recalculate();
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }
  if (running) {
    digitalWrite(PULSE_PIN, HIGH);
    delayMicroseconds(50);
    digitalWrite(PULSE_PIN, LOW);
    delayMicroseconds(stepPeriodUs);
  }
}