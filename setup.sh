read -p "Please enter the mongodb host for the experiment-database informat <url:port>: " host
read -p "Please enter the database-name on this host: " dbname
read -p "Please enter the username for the database: " dbuser
read -p "And the user's password: " dbpwd

echo "EXPERIMENT_DB_HOST = '$host'\nEXPERIMENT_DB_NAME = '$dbname'\nEXPERIMENT_DB_USER = '$dbuser'\nEXPERIMENT_DB_PWD = '$dbpwd'\n" > ./xview/settings.py

read -p "Please enter a directory path that should be used to store and access data: " basedir
echo "DATA_BASEPATH = '$basedir'\n" >> ./xview/settings.py
