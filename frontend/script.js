new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
        labels: ['Владимир', 'Новгород', 'Полоцк', 'еще регион'],
        datasets: [{
            label: 'Экономика регионов',
            data: [5, 10, 7, 3],
            backgroundColor: ['red', 'blue', 'green', 'yellow']
        }]
    },
});

new Chart(document.getElementById('pieChart'), {
    type: 'pie',
    data: {
        labels: ['тест1', 'тест2', 'тест3', 'тест4'],
        datasets: [{
            label: 'Торговля регионов',
            data: [5, 10, 7, 3],
            backgroundColor: ['red', 'pink', 'green', 'yellow']
        }]
},
options: {
    plugins: {
        title: {
            display: true,
            text: 'Торговля регионов' 
        }
    }
}
});