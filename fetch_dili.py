import time
from tdc.single_pred import Tox

max_attempts = 100
for attempt in range(max_attempts):
    try:
        data = Tox(name='DILI')
        df = data.get_data()
        print(f"Success on attempt {attempt+1}")
        df.to_csv('dili_raw.csv', index=False)
        break
    except BaseException as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(300)
else:
    print("Gave up after max attempts")