-- Consultas de verificación para correr en pgAdmin (Query Tool) sobre la base vivienda_estudiantil.
-- Ejecuta una a la vez: selecciónala y presiona F5.

-- 1) ¿Cuántas filas tiene cada tabla? (debe coincidir con lo que imprimió Python)
SELECT 'viviendas' AS tabla, count(*) FROM viviendas
UNION ALL SELECT 'universidades', count(*) FROM universidades
UNION ALL SELECT 'seguridad_unidades', count(*) FROM seguridad_unidades
UNION ALL SELECT 'municipios', count(*) FROM municipios
UNION ALL SELECT 'comunas', count(*) FROM comunas
UNION ALL SELECT 'barrios', count(*) FROM barrios;

-- 2) Historial de cargas (cada vez que se renuevan los datos queda registrado)
SELECT * FROM cargas ORDER BY id DESC;

-- 3) Una búsqueda como la que hará la web: apartaestudios o apartamentos de hasta $2.000.000,
--    con al menos 1 habitación, ordenados por cercanía a la universidad más cercana
SELECT id, municipio, barrio, tipo_inmueble, precio, area_m2, habitaciones,
       universidad_cercana, round(dist_universidad_min_km::numeric, 2) AS km_universidad,
       indice_seguridad
FROM viviendas
WHERE precio <= 2000000
  AND lower(tipo_inmueble) LIKE 'aparta%'   -- apartaestudio(s) o apartamento(s), sin importar mayúsculas
ORDER BY dist_universidad_min_km
LIMIT 20;

-- 4) Unión entre tablas (JOIN): vivienda + datos de seguridad de su comuna/municipio
SELECT v.id, v.municipio, v.precio, s.nombre AS unidad, s.indice_seguridad, s.homicidios
FROM viviendas v
JOIN seguridad_unidades s ON s.unidad = v.unidad_seguridad
LIMIT 20;

-- 5) Solo con PostGIS: viviendas a menos de 1 km de la USB (sigla según config/universidades.csv)
--    En pgAdmin, la pestaña "Geometry Viewer" del resultado las dibuja en un mapa.
SELECT v.id, v.precio, v.barrio, v.geom,
       round((ST_Distance(v.geom::geography, u.geom::geography))::numeric) AS metros
FROM viviendas v
JOIN universidades u ON u.sigla = 'USB'
WHERE ST_DWithin(v.geom::geography, u.geom::geography, 1000)
ORDER BY metros;

-- 6) Solo con PostGIS: cada vivienda dentro de su polígono de comuna (cruce espacial en SQL)
SELECT c.nombre, count(*) AS viviendas
FROM viviendas v
JOIN comunas c ON ST_Within(v.geom, c.geom)
GROUP BY c.nombre ORDER BY viviendas DESC;
