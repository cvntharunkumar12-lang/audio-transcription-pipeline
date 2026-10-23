@echo off
echo ===== Making test files =====
ffmpeg -loglevel error -y -i sample.m4a sample.wav
ffmpeg -loglevel error -y -i sample.m4a sample.mp3
ffmpeg -loglevel error -y -i sample.m4a sample.ogg
echo.
echo ===== TEST 1: WAV =====
python transcriber.py sample.wav --model base
echo.
echo ===== TEST 2: MP3 =====
python transcriber.py sample.mp3 --model base
echo.
echo ===== TEST 3: OGG =====
python transcriber.py sample.ogg --model base
echo.
echo ===== TEST 4: FAKE FILE (should show a clean error) =====
echo not audio > fake.mp3
python transcriber.py fake.mp3
echo.
echo ===== ALL TESTS DONE =====
