rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

python setup.py sdist bdist_wheel
python -m twine upload -u wangye_hope -p haliluya314159 --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*

pause

rd /s /Q build
rd /s /Q dist
rd /s /Q pywayne.egg-info

pip uninstall pywayne
pip install -i https://test.pypi.org/simple/ pywayne

pause