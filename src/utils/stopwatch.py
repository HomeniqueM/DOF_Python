import time as Time
# Codigo utilizado no semestre passado (Homenique)


class Stopwatch():

    def __init__(self) -> None:
        self.__end   = 0
        self.__start = 0

    def get_time(self) -> str:
        return "{:.2f}".format(self.__end - self.__start)

    def start(self):
        self.__start = Time.perf_counter()

    def stop(self) -> str:
        self.__end = Time.perf_counter()
        