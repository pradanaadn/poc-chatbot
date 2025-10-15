from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class QdrantConfig(BaseSettings):
    api_key: SecretStr = Field(json_schema_extra="QDRANT_API_KEY")
    host: str = Field("localhost", json_schema_extra="QDRANT_HOST")
    http_port: int = Field(6333, json_schema_extra="QDRANT_HTTP_PORT", alias="port")
    collection_name: str = Field("documents", json_schema_extra="QDRANT_COLLECTION_NAME")
    vector_size: int = Field(1536, json_schema_extra="QDRANT_VECTOR_SIZE")
    distance: str = Field("Cosine", json_schema_extra="QDRANT_DISTANCE")
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  
        extra="ignore",  
        populate_by_name=True
    )

class GeminiConfig(BaseSettings):
    api_key: SecretStr = Field(json_schema_extra="GEMINI_API_KEY")
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  
        extra="ignore",  
    )

class AppConfig(BaseSettings):
    http_host: str = Field("0.0.0.0", json_schema_extra="HTTP_HOST")
    http_port: int = Field(8000, json_schema_extra="HTTP_PORT")

    qdrant: QdrantConfig = QdrantConfig()
    gemini: GeminiConfig = GeminiConfig()
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  
    )
    
app_config = AppConfig()
