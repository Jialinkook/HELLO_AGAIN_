#include <Servo.h>

// ============================================================
// PUPPET EXHIBITION BUILD 5 package / BUILD 4-compatible firmware
// One servo on Arduino Mega D9, powered by the external 5V supply.
// External supply GND must also connect to Arduino GND.
//
// Python sends one line per response:
//   A180
//   A156
//   A120
//   B180  (one deliberately dramatic second-wind double hit)
//   R     (immediate reset to 0 degrees)
//   V     (return firmware identity for Python verification)
//
// W, T, w or t can still be used as a full 180-degree manual test.
// ============================================================

Servo puppetServo;

const byte SERVO_PIN = 9;
const int REST_ANGLE = 0;
const int FULL_LIFT_ANGLE = 180;
// Keep this identity stable so an already-working BUILD 4 upload remains
// compatible with the new Python interface. The servo signal stays on D9.
const char FIRMWARE_ID[] = "PUPPET_BUILD_4";

// Keep the same tested action timings.
const unsigned long LIFT_HOLD_MS = 650;
const unsigned long RETURN_HOLD_MS = 500;

// The burst happens only once in each ten-response audience cycle. It snaps
// high, drops briefly, snaps high again, then returns to rest. All timings are
// non-blocking so Python can still read the physical motion state.
const int BURST_DIP_ANGLE = 30;
const unsigned long BURST_FIRST_HIGH_MS = 600;
const unsigned long BURST_DIP_MS = 320;
const unsigned long BURST_SECOND_HIGH_MS = 700;


enum MotionState {
  IDLE,
  HOLDING_LIFT,
  HOLDING_RETURN,
  BURST_FIRST_HIGH,
  BURST_DIP,
  BURST_SECOND_HIGH
};

MotionState motionState = IDLE;
unsigned long stateStartTime = 0;
int activeLiftAngle = FULL_LIFT_ANGLE;

// Buffer for commands such as A156 followed by a newline.
char commandBuffer[8];
byte commandLength = 0;


void startWave(int requestedAngle) {
  if (motionState != IDLE) {
    Serial.println("BUSY");
    return;
  }

  activeLiftAngle = constrain(requestedAngle, 0, 180);

  Serial.print("WAVE_START:A");
  Serial.println(activeLiftAngle);

  // The movement begins immediately. There is no delay before this write.
  puppetServo.write(activeLiftAngle);
  stateStartTime = millis();
  motionState = HOLDING_LIFT;
}


void startBurst(int requestedAngle) {
  if (motionState != IDLE) {
    Serial.println("BUSY");
    return;
  }

  activeLiftAngle = constrain(requestedAngle, 0, 180);

  Serial.print("BURST_START:B");
  Serial.println(activeLiftAngle);

  puppetServo.write(activeLiftAngle);
  stateStartTime = millis();
  motionState = BURST_FIRST_HIGH;
}


void resetMotion() {
  puppetServo.write(REST_ANGLE);
  activeLiftAngle = FULL_LIFT_ANGLE;
  stateStartTime = millis();
  motionState = IDLE;
  Serial.println("RESET_DONE");
}


void updateWave() {
  const unsigned long currentTime = millis();

  if (
    motionState == HOLDING_LIFT &&
    currentTime - stateStartTime >= LIFT_HOLD_MS
  ) {
    puppetServo.write(REST_ANGLE);
    stateStartTime = currentTime;
    motionState = HOLDING_RETURN;
  }
  else if (
    motionState == HOLDING_RETURN &&
    currentTime - stateStartTime >= RETURN_HOLD_MS
  ) {
    motionState = IDLE;
    Serial.println("WAVE_DONE");
  }
  else if (
    motionState == BURST_FIRST_HIGH &&
    currentTime - stateStartTime >= BURST_FIRST_HIGH_MS
  ) {
    puppetServo.write(BURST_DIP_ANGLE);
    stateStartTime = currentTime;
    motionState = BURST_DIP;
  }
  else if (
    motionState == BURST_DIP &&
    currentTime - stateStartTime >= BURST_DIP_MS
  ) {
    puppetServo.write(activeLiftAngle);
    stateStartTime = currentTime;
    motionState = BURST_SECOND_HIGH;
  }
  else if (
    motionState == BURST_SECOND_HIGH &&
    currentTime - stateStartTime >= BURST_SECOND_HIGH_MS
  ) {
    puppetServo.write(REST_ANGLE);
    stateStartTime = currentTime;
    motionState = HOLDING_RETURN;
  }
}


void processCommand() {
  commandBuffer[commandLength] = '\0';

  if (
    commandLength >= 2 &&
    (commandBuffer[0] == 'A' || commandBuffer[0] == 'a')
  ) {
    const int requestedAngle = atoi(commandBuffer + 1);
    startWave(requestedAngle);
  }
  else if (
    commandLength >= 2 &&
    (commandBuffer[0] == 'B' || commandBuffer[0] == 'b')
  ) {
    const int requestedAngle = atoi(commandBuffer + 1);
    startBurst(requestedAngle);
  }
  else if (
    commandLength == 1 &&
    (commandBuffer[0] == 'R' || commandBuffer[0] == 'r')
  ) {
    resetMotion();
  }
  else if (
    commandLength == 1 &&
    (commandBuffer[0] == 'V' || commandBuffer[0] == 'v')
  ) {
    Serial.print("FIRMWARE:");
    Serial.println(FIRMWARE_ID);
  }
  else if (
    commandLength == 1 &&
    (
      commandBuffer[0] == 'W' || commandBuffer[0] == 'w' ||
      commandBuffer[0] == 'T' || commandBuffer[0] == 't'
    )
  ) {
    startWave(FULL_LIFT_ANGLE);
  }

  commandLength = 0;
}


void readSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = Serial.read();

    if (incoming == '\n' || incoming == '\r') {
      if (commandLength > 0) {
        processCommand();
      }
      continue;
    }

    // A single W or T remains convenient in Arduino Serial Monitor even when
    // its line-ending setting is "No line ending".
    if (
      commandLength == 0 &&
      (incoming == 'W' || incoming == 'w' || incoming == 'T' || incoming == 't')
    ) {
      commandBuffer[0] = incoming;
      commandLength = 1;
      processCommand();
      continue;
    }

    if (commandLength < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLength] = incoming;
      commandLength++;
    }
    else {
      // Discard an invalid overlong command safely.
      commandLength = 0;
    }
  }
}


void setup() {
  Serial.begin(9600);

  puppetServo.attach(SERVO_PIN);
  puppetServo.write(REST_ANGLE);

  // Python waits while the Arduino resets after opening the serial port, so
  // setup does not need a blocking delay here.
  Serial.print("READY:");
  Serial.println(FIRMWARE_ID);
}


void loop() {
  readSerialCommands();
  updateWave();
}
