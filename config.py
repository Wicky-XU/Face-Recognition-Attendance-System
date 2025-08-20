"""
人脸识别考勤系统配置文件
Face Recognition Attendance System Configuration

"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.absolute()

# =============================================================================
# Flask Web应用配置
# =============================================================================

class Config:
    """基础配置类"""
    
    # Flask基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    DEBUG = False
    TESTING = False
    
    # 服务器配置
    HOST = '0.0.0.0'
    PORT = 5000
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = BASE_DIR / 'uploads'  # 临时上传目录
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    HOST = '127.0.0.1'

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True

# 根据环境变量选择配置
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# =============================================================================
# 人脸识别模型配置
# =============================================================================

class FaceRecognitionConfig:
    """人脸识别相关配置"""
    
    # 模型文件路径
    MODEL_PATH = BASE_DIR / 'models' / 'facenet_lfw.pth'
    BACKUP_MODEL_PATH = BASE_DIR / 'training' / 'scripts' / 'models' / 'facenet_lfw.pth'
    
    # MTCNN人脸检测配置
    MTCNN_IMAGE_SIZE = 160
    MTCNN_MARGIN = 0
    MTCNN_KEEP_ALL = False
    MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net阈值
    
    # FaceNet特征提取配置
    EMBEDDING_SIZE = 512
    IMAGE_SIZE = (160, 160)
    
    # 识别判断配置
    RECOGNITION_THRESHOLD = 1.2  # 欧氏距离阈值，越小越严格
    MAX_FACES_PER_FRAME = 5
    MIN_FACE_SIZE = 20
    
    # 图像预处理配置
    NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    NORMALIZE_STD = [0.5, 0.5, 0.5]
    
    # 设备配置
    USE_GPU = True
    DEVICE = 'cuda' if USE_GPU else 'cpu'

# =============================================================================
# 数据库配置
# =============================================================================

class DatabaseConfig:
    """数据库配置"""
    
    # MySQL配置（主要使用）
    MYSQL_HOST = os.environ.get('MYSQL_HOST') or 'localhost'
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT') or 3306)
    MYSQL_USER = os.environ.get('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD') or ''
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE') or 'attendance_system'
    
    # SQLite配置（测试环境备用）
    SQLITE_DB_PATH = BASE_DIR / 'attendance.db'
    
    # 数据库类型选择
    DATABASE_TYPE = os.environ.get('DATABASE_TYPE') or 'mysql'  # 'mysql' 或 'sqlite'
    
    # 连接池配置
    POOL_SIZE = 10
    MAX_OVERFLOW = 20
    POOL_TIMEOUT = 30
    POOL_RECYCLE = 3600

# =============================================================================
# 训练配置
# =============================================================================

class TrainingConfig:
    """模型训练配置"""
    
    # 数据集路径
    DATASET_PATH = BASE_DIR / 'training' / 'dataset'
    LFW_PATH = DATASET_PATH / 'lfw-deepfunneled'
    METADATA_PATH = DATASET_PATH / 'metadata'
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 1e-4
    
    # 优化器配置
    OPTIMIZER = 'adam'
    MOMENTUM = 0.9
    BETA1 = 0.9
    BETA2 = 0.999
    
    # 学习率调度
    LR_SCHEDULER = 'step'
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.1
    
    # 早停配置
    EARLY_STOPPING = True
    PATIENCE = 5
    MIN_DELTA = 0.001
    
    # 模型保存
    SAVE_MODEL_PATH = BASE_DIR / 'training' / 'scripts' / 'models'
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINT_FREQ = 5

# =============================================================================
# 日志配置
# =============================================================================

class LoggingConfig:
    """日志配置"""
    
    # 日志目录
    LOG_DIR = BASE_DIR / 'logs'
    
    # 日志级别
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    
    # 日志格式
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # 日志文件配置
    LOG_FILE = LOG_DIR / 'attendance_system.log'
    ERROR_LOG_FILE = LOG_DIR / 'error.log'
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

# =============================================================================
# 安全配置
# =============================================================================

class SecurityConfig:
    """安全相关配置"""
    
    # 会话配置
    SESSION_TIMEOUT = 3600  # 1小时
    PERMANENT_SESSION_LIFETIME = 7200  # 2小时
    
    # CSRF保护
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    
    # 文件安全
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    # 限流配置
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URL = "memory://"

# =============================================================================
# 工具函数
# =============================================================================

def get_config():
    """获取当前环境配置"""
    env = os.environ.get('FLASK_ENV') or 'default'
    return config[env]

def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        BASE_DIR / 'uploads',  # 临时上传目录
        BASE_DIR / 'logs',
        BASE_DIR / 'models',
        TrainingConfig.SAVE_MODEL_PATH,
        LoggingConfig.LOG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查模型文件
    if not FaceRecognitionConfig.MODEL_PATH.exists():
        if not FaceRecognitionConfig.BACKUP_MODEL_PATH.exists():
            errors.append(f"模型文件不存在: {FaceRecognitionConfig.MODEL_PATH}")
    
    # 检查数据库配置
    if DatabaseConfig.DATABASE_TYPE not in ['sqlite', 'mysql']:
        errors.append(f"不支持的数据库类型: {DatabaseConfig.DATABASE_TYPE}")
    
    # 检查MySQL连接配置（如果使用MySQL）
    if DatabaseConfig.DATABASE_TYPE == 'mysql':
        if not DatabaseConfig.MYSQL_HOST:
            errors.append("MySQL主机地址未配置")
        if not DatabaseConfig.MYSQL_DATABASE:
            errors.append("MySQL数据库名未配置")
    
    # 检查阈值合理性
    if not 0.1 <= FaceRecognitionConfig.RECOGNITION_THRESHOLD <= 3.0:
        errors.append(f"识别阈值超出合理范围: {FaceRecognitionConfig.RECOGNITION_THRESHOLD}")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    return True

# 初始化配置
if __name__ == "__main__":
    ensure_directories()
    validate_config()
    print("配置验证通过")
else:
    ensure_directories()