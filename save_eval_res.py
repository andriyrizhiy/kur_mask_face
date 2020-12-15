from lib.helper.logger import logger
from lib.core.base_trainer.net_work import trainner
import setproctitle

logger.info('eval start')
setproctitle.setproctitle("detect")

trainner = trainner()

trainner.save_val_result()
