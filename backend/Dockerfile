
# FROM python:alpine
# Use the python:3.9image
# 指定一個 image 來源
FROM python:3.10.16

# Set the working directory to /app
WORKDIR /backend

# Copy the current directory contents into the container at /app 
# ADD：將檔案加到 images 內
ADD . /backend

# Install the dependencies
# build 時使用，會執行此命令
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]


# ENV：環境變數設定