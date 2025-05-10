@echo off
call C:\Users\user\anaconda3\Scripts\activate.bat HDNetAPP
cd /d C:\Users\user\Desktop\HDNetAPP
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
