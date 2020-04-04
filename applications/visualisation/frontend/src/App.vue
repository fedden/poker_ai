<template>
<div class="vue-container">
    <div class="table">
        <div class="card-place">
            <card v-for="(card, index) in five_cards" :card="card"></card>
        </div>
        <div class="players">
            <div v-for="(value, index) in players"
                  class="player" 
                  :class="['player-' + (index + 1), {'playing': player_playing === index}]">
                <div class="bank">
                    <div class="bank-value">{{ value.bank - value.onTable }}</div>
                    <div class="jetons v-10" v-if="(value.bank - value.onTable) / 10 >= 1"></div>
                    <div class="jetons v-5" v-if="(value.bank - value.onTable) / 2 >= 1"></div>
                    <div class="jetons v-1" v-if="(value.bank - value.onTable) >= 1"></div>
                </div>
                <div class="avatar" :style="{backgroundColor: value.color || 'dodgerblue'}"></div>
                <div class="name">{{value.name}}</div>
                <div class="mise">
                    <div class="mise-value">
                        {{ value.onTable }}
                    </div>
                    <div class="jeton-10">    
                        <div class="jetons v-10" v-for="(n, i) in ((value.onTable - (value.onTable % 10)) / 10)" :style="{top: (-2 + i) + 'px'}" v-if="value.onTable / 10 >= 1"></div>
                    </div>
                    <div class="jeton-5">
                        <div class="jetons v-5" v-for="(n, i) in (((value.onTable % 10) - ((value.onTable % 10) % 2)) / 2)" :style="{top: (-2 + i) + 'px'}" v-if="value.onTable % 10 && value.onTable % 10 >= 2"></div>
                    </div>
                    <div class="jeton-1">
                        <div class="jetons v-1" v-if="value.onTable % 10 && value.onTable % 2"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <button class="bouton" @click="">Refresh</button>
</div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>

html, body{
    margin:0;
    padding:0;
}

.vue-container{
    width:100vw;
    height:100vh;
}

.bank{
    position:absolute;
    width:30px;
    height:30px;
    top:100px;
    right:0;
    z-index:10;
    .bank-value{
        position:absolute;
        font-family:"Houschka Rounded";
        right:-30px;
        font-weight:bold;
        font-size:12px;
        color:#96e296;
        &:after{
            content:"$";
        }
    }
    .v-10{
        top:-2px;
        right:-2px;
        background-color:crimson;
    }
    .v-5{
        top:15px;
        right:5px;
        background-color:goldenrod;
    }
    .v-1{
        top:22px;
        left:-10px;
        background-color:#444;
    }
}

.mise{
    position:absolute;
    top:170px;
    left:50%;
    transform:translatex(-50%);
    height:40px;
    width:60px;
    .mise-value{
        position:absolute;
        font-family:"Houschka Rounded";
        right:-30px;
        font-weight:bold;
        font-size:12px;
        color:#96e296;
        &:after{
            content:"$";
        }
    }
    [class^="jeton-"]{    
        position:absolute;
    }
    .jeton-10{
        top:-2px;
        left:0px;
    }
    .jeton-5{
        top:-2px;
        left:22px;
        background-color:goldenrod;
    }
    .jeton-1{
        top:-2px;
        right:15px;
        background-color:#444;
    }
}


.jetons{
    position:absolute;
    width:12px;
    height:12px;
    border-radius:100%;
    border:2px white dotted;
    box-shadow:2px 2px 0 rgba(0, 0, 0, .1);
    &.v-10{
        background-color:crimson;
    }
    &.v-5{
        background-color:goldenrod;
    }
    &.v-1{
        background-color:#444;
    }
}

.card{
    height:70px;
    width:50px;
    border-radius:5px;
    display:inline-block;
    position:relative;
    background-color:white;
    font-family:"Houschka Rounded";
    &.figures-P, &.figures-C{
        color:#515260;
    }
    &.figures-H, &.figures-D{
        color:#EC6E69;
    }
    h1{
        font-size:10px;
        text-align:center;
        position:absolute;
        margin:0;
        &:first-child{
            top:5px;
            left:5px;
        }
        &:last-child{
            bottom:5px;
            right:5px;
            transform:rotatez(180deg)
        }
    }
    .figures{
        position:absolute;
        top:50%;
        left:50%;
        transform:translateX(-50%) translateY(-50%);
    }
}

.figures{
    width:20px;
    height:20px;
    background-image:url('https://s3-us-west-2.amazonaws.com/s.cdpn.io/441095/clovers.svg');
    background-size:40px;
    &.P{
        background-position:0 0;
    }
    &.D{
        background-position:20px 0;
    }
    &.C{
        background-position:20px 20px;
    }
    &.H{
        background-position:0 20px;
    }
}

.table{
    width:1000px;
    height:400px;
    background-color:#4aad4a;
    position:absolute;
    left:50%;
    top:50%;
    transform:translateX(-50%) translateY(-50%);
    border-radius:150px;
    position:relative;
    border:15px solid #a95555;
    &:before{
        content:'';
        border:7px solid rgba(0, 0, 0, .1);
        display:block;
        width:1015px;
        height:415px;
        border-radius:150px;
        position:absolute;
        top:-15px;
        left:-15px;
    }
    &:after{
        content:'';
        border:7px solid rgba(0, 0, 0, .1);
        display:block;
        width:985px;
        height:385px;
        border-radius:130px;
        position:absolute;
        top:0;
        left:0;
    }
    .card-place{
        border:5px solid #63c763;
        height:100px;
        width:340px;
        position:absolute;
        border-radius:10px;
        padding:10px;
        top:50%;
        left:50%;
        transform:translateX(-50%) translateY(-50%);
        box-sizing:border-box;
        .card:not(:last-child){
            margin-right:15px;
        }
    }
}

.players{
    position:relative;
    width:100%;
    height:100%;
    z-index:100;
    .player{
        position:absolute;
        .avatar{
            width:120px;
            height:120px;
            background-color:lightcoral;
            border-radius:100%;
            position:relative;
            box-shadow: 2px 10px 0px rgba(0, 0, 0, .4);
            z-index:20;
            &:after{
                content:'';
                width:70px;
                height:70px;
                position:absolute;
                background-color:rgba(0, 0, 0, .1);
                top:50%;
                left:50%;
                transform:translatex(-50%) translatey(-50%);
                border-radius:100%;
                box-shadow:0px 5px 0px rgba(0, 0, 0, .2)
            }
        }
        .name{
            font-family:"Houschka Rounded";
            text-align:center;
            width:100px;
            color:#96e296;
            padding:5px 0;
            margin-left:10px;
            box-sizing:border-box;
            border:2px solid #96e296;
            border-radius:5px;
            margin-top:15px;
            text-overflow:ellipsis;
            font-size:11px;
            overflow:hidden;
            position:relative;
        }
        &.playing:before{
            content:'...';
            color:white;
            font-size:20px;
            position:absolute;
            background-color:#76daff;
            display:inline-block;
            line-height:0px;
            height:10px;
            padding:5px 10px;
            border-radius:5px;
            z-index:100;
        }
        &.player-1{
            top:0px;
            left:50%;
            transform:translatex(-50%) translatey(-50%);
        }
        &.player-2{
            bottom:0px;
            left:50%;
            transform:translatex(-50%) translatey(50%) rotatez(180deg);
            .name{
                transform:rotatez(180deg);
            }
            .bank-value{
                transform:rotatez(180deg);
            }
            .mise-value{
                transform:rotatez(180deg);
            }
        }
        &.player-3{
            top:50%;
            left:0px;    
            transform:translatex(-50%) translatey(-50%) rotatez(-90deg);
            .name{
                transform:rotatez(0deg);
            }
        }
        &.player-4{
            top:50%;
            right:0px;    
            transform:translatex(50%) translatey(-50%) rotatez(90deg);
            .name{
                transform:rotatez(0deg);
            }
        }
        &.player-5{
            top:0px;
            left:25%;
            transform:translatex(-50%) translatey(-50%);
        }
        &.player-6{
            bottom:0px;
            left:75%;
            transform:translatex(-50%) translatey(50%) rotatez(180deg);
            .name{
                transform:rotatez(180deg);
            }
            .bank-value{
                transform:rotatez(180deg);
            }
            .mise-value{
                transform:rotatez(180deg);
            }
        }
        &.player-7{
            top:0px;
            left:75%;
            transform:translatex(-50%) translatey(-50%);
        }
        &.player-8{
            bottom:0px;
            left:25%;
            transform:translatex(-50%) translatey(50%) rotatez(180deg);
            .name{
                transform:rotatez(180deg);
            }
            .bank-value{
                transform:rotatez(180deg);
            }
            .mise-value{
                transform:rotatez(180deg);
            }
        }
    }
}

.bouton{
    background-color:#515260;
    color:white;
    text-transform:uppercase;
    border:none;
    outline:none;
    padding:5px 10px;
}
</style>
