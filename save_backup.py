import paramiko
import os


class SFTPUploader:
    def __init__(self, host, port, username, password=None, key_file=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_file = key_file
        self.transport = None
        self.sftp = None

    def connect(self):
        self.transport = paramiko.Transport((self.host, self.port))
        if self.key_file:
            private_key = paramiko.RSAKey.from_private_key_file(self.key_file)
            self.transport.connect(username=self.username, pkey=private_key)
        else:
            self.transport.connect(username=self.username, password=self.password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def upload(self, local_path, remote_path):
        if not self.sftp:
            self.connect()

        remote_dir = os.path.dirname(remote_path)
        self._mkdirs(remote_dir)
        self.sftp.put(local_path, remote_path)
        print(f"[SFTP] Uploaded {local_path} -> {remote_path}")

    def _mkdirs(self, remote_directory):
        dirs = remote_directory.strip('/').split('/')
        path = ''
        for d in dirs:
            path += f'/{d}'
            try:
                self.sftp.stat(path)
            except FileNotFoundError:
                self.sftp.mkdir(path)

    def close(self):
        if self.sftp:
            self.sftp.close()
        if self.transport:
            self.transport.close()
