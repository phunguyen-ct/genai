# Chatbot Real Estate renting consultant 

- Creating an Environment from a File
```
conda env create --file environment.yml
```

## Install dependencies

- Create env

```
conda create --name vietai python=3.12
```

- Install packages

```
pip install -r requirements.txt
```

## Required env variables

Open your terminal and run these below command

```
export GOOGLE_API_KEY=<Your OpenAI key>
```

## Start the app

```
python main.py
```

Notice: The verify time you run the app, if there is not any data collection, it's might takes a few minutes to crawl web data before you can chat

![Alt text](./demo.png "")
