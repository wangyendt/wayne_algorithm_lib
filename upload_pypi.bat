rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

python setup.py sdist bdist_wheel
twine upload dist/*

pause

rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

pause
