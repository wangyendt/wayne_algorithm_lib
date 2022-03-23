rm -rf build
rm -rf dist
rm -rf pywayne.egg-info

python38 setup.py sdist bdist_wheel
twine upload -u wangye_hope -p haliluya314159 dist/*

read -p "Press any key to resume ..."

rm -rf build
rm -rf dist
rm -rf pywayne.egg-info

pip uninstall pywayne -y
pip install -U -i https://pypi.org/simple pywayne

read -p "Press any key to resume ..."
