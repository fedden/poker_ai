import Vue from 'vue'
import App from './App'
import router from './router'
import SocketIO from 'socket.io-client'
import VueSocketIO from 'vue-socket.io'

Vue.use(
  new VueSocketIO({
    debug: true,
    connection: SocketIO('http://localhost:5000'),
  })
)
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
  sockets: {
    // Fired when the server sends something on the "state" channel.
    state(data) {
      this.player_playing = data.player_playing
      this.players = data.players
      this.five_cards = data.five_cards
    },
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
