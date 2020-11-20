rm -rf build
rm -rf dist
rm -rf pywayne.egg-info

python setup.py sdist bdist_wheel
python -m twine upload -u wangye_hope -p haliluya314159 --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*

read -p "Press any key to resume ..."

rm -rf build
rm -rf dist
rm -rf pywayne.egg-info

pip uninstall pywayne
pip install -i https://test.pypi.org/simple/ pywayne

read -p "Press any key to resume ..."
