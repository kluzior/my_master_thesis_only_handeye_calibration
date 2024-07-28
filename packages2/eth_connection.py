import socket



class EthConnection:
    def __init__(self, host="192.168.0.1", port=10000):
        self.host = host
        self.port = port


    def create_client(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.connect((self.host, self.port))
        print("Start listening...")
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        c, addr = self.socket.accept()

        return c


    def close(self):
        # self.socket.close()
        pass


# if __name__ == "__main__":
#     HOST = "192.168.0.1"
#     PORT = 10000
#     print("Start listening...")
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     s.bind((HOST, PORT))
#     s.listen(5)
#     c, addr = s.accept()

    
#     configure_logger("debug_handeye")
#     handeye = HandEyeCalibration(c)
#     a, positions  = handeye.run()
#     print(f"{len(a)}")
#     pos_list = [list(pos) for pos in positions]
#     print(pos_list)
#     handeye.prepare_data(a, pos_list)
#     print("Done!") 