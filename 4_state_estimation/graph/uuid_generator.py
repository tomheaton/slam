from os.path import exists


class UUIDGenerator():
    # TODO: Use __file__ to make sure it is always in the correct folder
    __last_uuid_file = 'last_uuid.dat'

    @classmethod
    def __get_last_uuid(cls):
        if not exists(cls.__last_uuid_file):
            return -1

        with open(cls.__last_uuid_file, 'r') as f:
            return int(f.readline().strip())

    @classmethod
    def __update_last_uuid(cls, uuid):
        with open(cls.__last_uuid_file, 'w') as f:
            f.write(str(uuid))

    @classmethod
    def generate_uuid(cls):
        last_uuid = cls.__get_last_uuid()
        next_uuid = last_uuid + 1
        cls.__update_last_uuid(next_uuid)

        return next_uuid
