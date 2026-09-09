# HELLO_AGAIN_
================================================

WHAT CHANGED
------------
- Camera capture, MediaPipe tracking and the interface are capped at 15 FPS.
  Camera input is 960x540 while the final exhibition display remains
  1920x1080, reducing laptop load without shrinking the projected image.
- The main instruction and PUPPET THOUGHT use a portable block-pixel typeface.
  It is drawn by the program and does not require a font installation.
- THANK YOU remains on screen for exactly five seconds after response 10, then
  the work returns automatically to the black-and-white start screen.
- When the upper-body appearance changes continuously for about one second,
  the current partial interaction is reset for the new visitor. This uses a
  temporary clothing/torso colour histogram, not face recognition. It does not
  save a body crop or biometric identity data.
- Before the first accepted wave, the live camera and memory field are black
  and white. The centre of the screen clearly says WAVE YOUR HANDS.
- The first accepted wave restores the camera's original colour immediately.
- During the visitor session, the centre instruction says KEEP WAVING.
- At response 10, the centre instruction says THANK YOU / STEP AWAY FOR THE
  NEXT PERSON. Response 10 no longer starts a water break.
- BODY ENERGY and the articulated puppet scan are significantly larger.
- PUPPET THOUGHT is a large, high-contrast lower-third line rendered with a
  TrueType font at the final 1920x1080 display resolution.
- Each visitor receives one persistent five-digit archive identifier:
  N.00001, N.00002, N.00003, and so on.
- One faint identifier line appears at the top of that visitor's memory
  window. Each visitor contributes one representative image, not one image
  per wave.
- The right panel shows VISITORS 01/10 for the current water-break cycle.
- A 30-second water break starts only after the tenth separate visitor session
  ends. It no longer starts after ten waves from one person.

HOW A VISITOR SESSION IS COUNTED
--------------------------------
1. A new session is registered when a visitor's first wave is accepted.
2. The visitor can perform up to ten wave responses.
3. After response 10, THANK YOU stays for five seconds and the screen returns
   to its black-and-white waiting state automatically.
4. An unfinished round also resets after four seconds with nobody visible, or
   after a sustained change to a different-looking visitor.
5. After visitor 10 completes the five-second THANK YOU screen, the 30-second
   water break begins.
6. After the break, VISITORS returns to 00/10. Archive numbering continues.

The system counts separate interaction sessions, not biometric identities.
The visitor-change feature compares clothing/torso appearance only, so two
people in very similar clothing may still require one person to step away.
Two people standing together are treated as one session.

RUN
---
1. Extract every file into one new empty folder.
2. Keep Arduino Serial Monitor closed.
3. Double-click START_PUPPET_BUILD_5_3.bat.

If BUILD 4 firmware already works with the servo signal on Arduino Mega D9,
you do not need to upload it again. PUPPET_BUILD_5_D9.ino is included only as
a matching backup.

INSTALL DEPENDENCIES IF NEEDED
------------------------------
py -m pip install -r PUPPET_BUILD_5_REQUIREMENTS.txt

VISITOR NUMBER FILE
-------------------
On first use, the program creates PUPPET_VISITOR_COUNTER.json beside the
Python file. This keeps the next N.00001-style number after the app closes.
To deliberately restart the archive numbering at N.00001, close the program
and delete only PUPPET_VISITOR_COUNTER.json.

CONTROLS
--------
N  Advance one response manually.
T  Test one ordinary 180-degree action.
B  Test the double second-wind burst.
C  Restart the current visitor round.
F  Toggle fullscreen.
H  Show/hide diagnostics.
Q  Quit.

SETTINGS
--------
At the top of PUPPET_EXHIBITION_BUILD_5_3.py:
SERIAL_PORT = "COM3"
CAMERA_INDEX = 1
TARGET_FPS = 15.0

Change only these values if Windows assigns a different device number.

EXPECTED TERMINAL LINE
----------------------
RUNNING BUILD 5.3: 15 FPS / 5S THANK YOU / VISITOR CHANGE

HARDWARE
--------
Servo signal: Arduino Mega D9.
Servo power: external 5V supply.
External power GND and Arduino GND must be connected together.
