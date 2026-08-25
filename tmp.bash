cat << 'EOF' > ~/.config/uv/uv.toml
allow-insecure-host = ["mirrors.aliyun.com", "download.pytorch.org"]

[pip]
index-url = "http://mirrors.aliyun.com/pypi/simple/"
EOF
