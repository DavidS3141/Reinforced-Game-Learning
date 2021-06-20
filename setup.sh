#!/usr/bin/env bash

source "$( dirname "${BASH_SOURCE[0]}" )/scripts/set_env_variables.bash"
if [ $? -ne 0 ]
then
    return 1
fi

# activate virtual env
if [ $VIRTUAL_ENV ]
then
    echo "deactivating previous venv"
    deactivate
fi

if [ -f "$SRC_DIR/.venv/bin/activate" ]
then
    :
else
    if (( $(python3.7 --version | cut -d. -f2) >= 6 )) # prompt option available since >=3.6
    then
        python3.7 -m venv "$SRC_DIR/.venv" --prompt="$PROJECT_SHORT"
    else
        python3.7 -m venv "$SRC_DIR/.venv"
    fi
fi

source "$SRC_DIR/.venv/bin/activate"

pip3 install -U pip
pip3 install -U -r $SRC_DIR/requirements.txt

# make sure pre-commit hooks are installed
(cd $SRC_DIR && pre-commit install)

# set project path so that project imports work
if [ -f "$VIRTUAL_ENV/lib/python3.*/site-packages/project_path.pth" ]
then
    (cd $VIRTUAL_ENV/lib/python3.*/site-packages && rm project_path.pth)
fi
(cd $VIRTUAL_ENV/lib/python3.*/site-packages && echo $SRC_DIR/src > project_path.pth)
(cd $VIRTUAL_ENV/lib/python3.*/site-packages && echo $SRC_DIR/packages >> project_path.pth)

# build all static compilation units
bash "$SRC_DIR/scripts/build_all.bash"
