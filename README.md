## SETUP
1. for mac/linux run these 3 commands - 
    ```
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

    for windows - 
    ```
    python -m venv env
    .\env\Scripts\activate
    pip install -r requirements.txt
    ```
    
2. run infra_services by going inside the folder and running the following command - 
    ```
    docker-compose up -d
    ```

3. Finally run the server
    ```
    python app.py
    ```