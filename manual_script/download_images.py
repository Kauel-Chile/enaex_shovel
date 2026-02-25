import asyncio
import httpx
from pathlib import Path

# Configuración
URLS_FILE = 'urls.txt'
OUTPUT_DIR = Path('images')
CONCURRENT_LIMIT = 5  # Número de descargas simultáneas

async def download_image(client, url, semaphore):
    async with semaphore:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            
            # Extraer nombre del archivo del URL o usar un fallback
            filename = url.split('/')[-1].split('?')[0] 
            if not filename: filename = f"image_{hash(url)}.jpg"
            
            file_path = OUTPUT_DIR / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Descargado: {filename}")
            
        except Exception as e:
            print(f"❌ Error con {url}: {e}")

async def main():
    # Crear carpeta si no existe
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Leer URLs
    if not Path(URLS_FILE).exists():
        print(f"Archivo {URLS_FILE} no encontrado.")
        return

    urls = [line.strip() for line in open(URLS_FILE) if line.strip()]
    
    # Semáforo para no saturar el servidor o tu conexión
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [download_image(client, url, semaphore) for url in urls]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())