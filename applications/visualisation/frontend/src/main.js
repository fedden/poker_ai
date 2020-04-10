import Vue from 'vue'
import App from './App'
import axios from 'axios'
import router from './router'

Vue.config.productionTip = false

new Vue({
  el: '#app',
  router,
  render: function (createElement) {
    return createElement(App, {
      props: {
        player_playing: this.player_playing,
        players: this.players,
        figures: this.figures,
        values: this.values,
        cards: this.cards,
        five_cards: this.five_cards,
      },
    })
  },
  methods: {
    getState() {
      const path = 'http://localhost:5000/api/state'
      axios
        .get(path)
        .then((response) => {
          this.player_playing = response.data.player_playing
          this.players = response.data.players
          this.five_cards = response.data.five_cards
        })
        .catch((error) => {
          console.log(error)
        })
    },
  },
  created() {
    this.getState()
  },
  data: {
    player_playing: 0,
    players: [],
    five_cards: [],
    figures: ['P', 'H', 'C', 'D'],
    values: ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'],
  },
  computed: {
    cards() {
      let all = []
      for (let figure of this.figures) {
        for (let value of this.values) {
          all.push({
            f: figure,
            v: value,
          })
        }
      }
      return all
    },
  },
})
