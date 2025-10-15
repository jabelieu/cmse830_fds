# README for midterm project
Welcome to my repo container for the semester project in CMSE 830 : Foundations of Data Science! Within this container should be all the necessary items to run the streamlit app code (midterm.py). I recommend this general process to get things going :

* Clone this repo locally.
* Create a virtual environment (example below)
    * python -m venv myenv_name
* Activate the environment and use pip to install the necessary versions of libraries (found in requirements.txt)
    * pip install -r requirements.txt
* Run the script!
    * streamlit run midterm.py

OR

Follow this link for the streamlit deployed app :

https://cmse830fds-vh3dcoi3b6nbvmjp8ds38a.streamlit.app/

This project makes use of the AME2020 dataset which is a collection of binding energy, beta decay energies, and related quantities of nuclei. It also uses the nuclear charge radii dataset which contains measurements of charge radii. Both datasets can be found on the International Atomic Energy Agency : Nuclear Data Services' website.