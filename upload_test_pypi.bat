rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

python setup.py sdist bdist_wheel
python -m twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*

pause

rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

pip install -U --index-url https://test.pypi.org/simple/ --no-deps pywayne

pause