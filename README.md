Instructions to execute code

1. You need to have python 3 installed in the system. 
   to check if python is installed in the system open command prompt(for windows) or terminal(for mac) and type 
	python --version
	if python version is not displayed then python is not installed most probably. 

To Install Python:
Windows
Follow the below steps to install Python 3 on Windows.

1. Go to the Python Releases for Windows page (https://www.python.org/downloads/windows/) and download the latest stable release Windows x86-64 executable installer.
2. After the download is complete, run the installer.
3. On the first page of the installer, be sure to select the “Add Python to PATH” option and click through the remaining setup steps leaving all the pre-select installation defaults.
4. Once complete, we can check that Python was installed correctly by opening a Command Prompt (CMD or PowerShell) and entering the command python --version. The latest Python 3.7 version number should print to the console.

Mac
1. Go to the Python Releases for Mac OS X (https://www.python.org/downloads/mac-osx/) page and download the latest stable release macOS 64-bit/32-bit installer.
2. After the download is complete, run the installer and click through the setup steps leaving all the pre-selected installation defaults.
3. Once complete, we can check that Python was installed correctly by opening a Terminal and entering the command python3 --version. The latest Python 3.7 version number should print to the Terminal.

2. Now that python is there in the system, we need to check for Jupyter Notebook
if Jupyter notebook is not installed we can easily install it using python by following the below steps

Follow the below instructions to install the Jupyter Notebook package using the pip Python package manager.

1. Open a new Terminal (Mac) or Command Prompt (Windows).
2. Run pip install jupyter to download and install the Jupyter Notebook package.
3. Once complete, we can check that Jupyter Notebook was successfully installed by running jupyter notebook from a Terminal (Mac) / Command Prompt (Windows). This will startup the Jupyter Notebook server, print out some information about the notebook server in the console, and open up a new browser tab at http://localhost:8888.

3. once we havre checked for both python and Jupyter notebook our next step in requirement is to install the libraries required to run the code 

the libraries needed to be installed are 
numpy==1.17.4
opencv-python==4.1.2.30
tensorflow==2.1.2
python==3.5.2

these are given in requirement.txt

use pip install -r requirements.txt to install the required libraries at once.

4. The last requirement is to install face_recognition library. This has an addition requirement of CMake and Dlib
To install CMake and Dblib you need visual studio(not vs code)
install visual studio using https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2414&workload=dotnetwebcloud&flight=FlipMacCodeCF;35d&installerFlight=FlipMacCodeCF;35d&passive=false#dotnet and set it up
after succesfull installation of visual studio run
  cmake .
  make      
  make install
to install CMake
and now we run 
pip install dlib and it gets installed successfully

5. Finally we need to install Face_recogntion using pip install face_recognition and we now have all the dependencies successfully installed.

6. Run untitled.ipynb in jupyter notebook to run the code.
