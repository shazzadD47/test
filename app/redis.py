from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
from redis.connection import ConnectionPool

from app.configs import settings

pool = ConnectionPool.from_url(str(settings.REDIS_URL), decode_responses=True)
langgraph_pool = AsyncConnectionPool.from_url(
    str(settings.REDIS_URL), decode_responses=False
)

redis_client = Redis(connection_pool=pool)
lg_redis_client = AsyncRedis(connection_pool=langgraph_pool)
