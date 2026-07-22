#container image. slim because I want only the necessary (no heavy and addition package)
FROM python:3.9-slim  
#create the working dir into the container
WORKDIR /app
# run and install all the package dependecy of the system
RUN pip install --no-cache-dir \ 
    fastapi==0.110.0 \
    uvicorn[standard]==0.28.0 \
    httpx==0.27.0 \
    python-multipart==0.0.9

COPY RestAPI.py .
#expose the contianer - process port ! notise that is the same expose by the container also
EXPOSE 5000
#this command start with the process container start 
CMD ["uvicorn", "RestAPI:app", "--host", "0.0.0.0", "--port", "5000"]
#uvicorn is a light web http server as tomcat or nodejs. use a single thread event loop (no cpu intensive operations)