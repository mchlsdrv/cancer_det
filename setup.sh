echo Creating the \'projects\' directory under the \'$HOME\' directory
mkdir $HOME/projects
cd $HOME/projects

echo Clonning the \'cancer_det\' project ...
git clone https://github.com/mchlsdrv/cancer_det.git

echo Creating the \'dev\' directory under the \'$HOME\' directory ...
mkdir $HOME/dev
cd $HOME/dev

echo Clonning the \'ml\' project to the \'$HOME/dev\' ...
git clone https://github.com/mchlsdrv/ml.git

echo Clonning the \'python_utils\' project to the \'$HOME/dev\' ...
git clone https://github.com/mchlsdrv/python_utils.git

echo Adding the path to \'$HOME/dev\' to the \'$PATH\'...
export PATH="$HOME/dev:$PATH"
echo * New path: \'$PATH\'

echo Adding the path to \'$HOME/dev/ml\' to the \'$PATH\' ...
export PATH="$HOME/dev/ml:$PATH"
echo * New path: \'$PATH\'

echo Adding the path to \'$HOME/dev/python_utils\' to the \'$PATH\' ...
export PATH="$HOME/dev/python_utils:$PATH"
echo * New path: \'$PATH\'

echo Saving the path $PATH to \'$HOME/.bashrc...
echo PATH="$PATH" >> ~/.bashrc

cd $HOME/projects/cancer_det
echo Done !
