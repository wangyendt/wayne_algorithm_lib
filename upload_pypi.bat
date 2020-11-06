rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

python setup.py sdist bdist_wheel
twine upload -u wangye_hope -p haliluya314159 dist/*

pause

rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

pip uninstall pywayne -y

pause
