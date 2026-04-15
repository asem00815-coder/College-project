// Функция для вычисления евклидова расстояния между двумя точками
function distance(x1, y1, x2, y2) {
    const dx = x1 - x2;
    const dy = y1 - y2;
    return Math.sqrt(dx * dx + dy * dy);
}

// Упрощение координат: удаляем точки, слишком близкие к предыдущей
function simplifyCoords(coords, threshold = 10) {
    if (coords.length < 4) return coords.slice(); // мало точек – нечего упрощать

    const result = [coords[0], coords[1]]; // первая точка всегда остаётся
    let lastX = coords[0];
    let lastY = coords[1];

    for (let i = 2; i < coords.length; i += 2) {
        const x = coords[i];
        const y = coords[i + 1];
        if (distance(lastX, lastY, x, y) >= threshold) {
            result.push(x, y);
            lastX = x;
            lastY = y;
        }
    }
    return result;
}

// Применяем упрощение ко всем регионам
const simplifiedRegions = rawRegions.map(region => ({
    ...region,
    rawCoords: simplifyCoords(region.rawCoords, 10) // порог 10, можно менять
}));

// Посмотрим результат для примера (регион Novgorod был очень детальным)
console.log(simplifiedRegions.find(r => r.name === "Novgorod").rawCoords.length);
// Было 216 чисел (108 точек), стало значительно меньше