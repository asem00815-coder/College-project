new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Экономика регионов', 
            data: [9, 7, 9],
            backgroundColor: ['red', 'blue', 'green']
        }]
    },
options: {
    plugins: {
        title: {
            display: true,
            text: 'Экономика регионов' 
        }
    }
}
});

new Chart(document.getElementById('pieChart'), {
    type: 'pie',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Торговля регионов в процентах',
            data: [2, 5, 3],
            backgroundColor: ['red', 'blue', 'green']
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
new Chart(document.getElementById('bar2Chart'), {
    type: 'bar',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Социальная свобода', 
            data: [3, 10, 5],
            backgroundColor: ['red', 'blue', 'green']
        }]
    },
options: {
    plugins: {
        title: {
            display: true,
            text: 'Экономика регионов' 
        }
    }
}
});
new Chart(document.getElementById('pie2Chart'), {
    type: 'pie',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Отдельная мощь регионов',
            data: [9, 5, 9],
            backgroundColor: ['red', 'blue', 'green']
        }]
},
options: {
    plugins: {
        title: {
            display: true,
            text: 'Отдельная мощь регионов'
        }
    }
}});