# check if venv already exists

if [ -d "venv" ]; then
    echo "venv exists"
else 
    echo "venv does not exist"
    echo "creating venv"
    virtualenv venv -p python3.8
    pip install -r requirements.txt
    python -m ipykernel install --user --name=venv
fi

source venv/bin/activate

while getopts ":u" option; do
   case $option in
      u) # update with libraries
        echo "updating venv"
        pip install -r requirements.txt
        python -m ipykernel install --user --name=venv
   esac
done

echo "activating venv"
source venv/bin/activate
