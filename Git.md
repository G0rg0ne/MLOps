To set the user config for any git repository already created : 
```bash
  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

```
To create a key-gen for your repo:
```shell
 ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```
