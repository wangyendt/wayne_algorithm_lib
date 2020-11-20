rm -rf build
rm -rf dist
rm -rf pywayne.egg-info

python setup.py sdist bdist_wheel
twine upload -u wangye_hope -p haliluya314159 dist/*

read -p "Press any key to resume ..."

rm -rf build
rm -rf dist
rm -rf pywayne.egg-info

pip uninstall pywayne -y

read -p "Press any key to resume ..."
