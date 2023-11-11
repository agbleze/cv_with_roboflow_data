#!/usr/bin/env bas

cat vscode-extensions.txt | while read extension || [[ -n $extension ]];
do
  code --install-extension $extension --force
doneh