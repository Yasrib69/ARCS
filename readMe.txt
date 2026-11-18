for back end make sure to cd into backend
to run back end type in the following:
python -m venv venv
source venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart mediapipe opencv-python numpy
python -m uvicorn main:app --host 0.0.0.0 --port 8000

this will set up a virtual enviroment where the python is 
the correct kind. then whenever you need to run 
the back end just type the following
python -m uvicorn main:app --host 0.0.0.0 --port 8000

and press control c to quit

for front end cd to Front end and then cd into Facetracking
to run the front end just type the following
npx expo start

and press control c to quit

also in app.tsx you will need to change the local ip 
to be your computers