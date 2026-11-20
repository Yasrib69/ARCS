# ARCS capstone demo notes

## What is new
Speaker lock for several faces: the caption goes to the person whose lip movement matches the
sound, not just to whoever's mouth moves most. Plus speaker labels and colours, captions that
show loudness, a no-camera demo, and an accuracy test. All of it is in the repo and covered by tests.

## Before the presentation (5 minutes, on your laptop)
The sandbox could not run the real webcam, microphone, MediaPipe or Whisper, so check these once yourself:
1. `cd backend`, activate your venv, `python -m uvicorn arcs.server:app`, open http://localhost:8000 in Chrome.
2. Click **Start captioning** with **Speaker lock** Off (the default). It should caption exactly like your original.
3. Switch **Speaker lock** to On. Get a second face in view (a friend, or a phone playing a video of someone talking) and talk. Both faces
   should get a colour and label, and the lock bars should move. Watch whether the caption follows YOU.
4. Click **Try the demo (no camera)** as the fallback. It needs only the backend, no camera, mic or internet.
5. `python -m pytest` should end with all tests passing (see the note on flaky teardown below).

## Suggested 3-minute flow
1. Problem (30 s): with several people, a caption that just follows moving mouths gets stolen by listeners.
2. Demo (60 s): the demo room. Point at "Really talking / Speaker lock picks / Mouth movement only picks".
3. Idea (30 s): lip movement is matched to the sound; the face that follows the sound is the speaker.
4. Numbers (30 s): **Run accuracy test**. Say plainly these are simulated conversations.
5. Limits and next steps (30 s): real-world testing with DHH students, overlapping speech, mobile app.

## If something looks wrong
Leave Speaker lock Off: that is your original caption behaviour. Use the demo button for the feature.

## Be upfront about
- The accuracy numbers come from simulated conversations (about 79% vs 67% for mouth movement alone). They
  check the logic. They are not results from real rooms.
- The demo room is deliberately busy (listeners move their mouths a lot). In calm rooms the old method is
  nearly as good, and the accuracy table shows the scenarios where it wins.
- One speaker at a time; overlapping speech is not handled.
- The loudness styling is relative to how loud the person usually speaks.
- Your requirements say multi-speaker is future work; this is an early preview of it.

## Known test flakiness
Roughly 1 in 25 full test runs, one websocket test ends with a CancelledError while closing. It is
the test client force-cancelling the server task right after a disconnect, it happens in tests that
were already there, and re-running passes. It does not affect the running app.

## Note on the demo button
The demo always uses the speaker lock, so the Speaker lock switch can show Off while the demo runs. That is expected.
