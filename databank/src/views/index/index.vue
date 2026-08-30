<template>
  <div class="home-container">
    <div class="introduce">
      <img class="logo" src="@/assets/images/logo-color.png" alt="logo" />
      <div class="title">{{ $t("introduce.title") }}</div>
      <div class="content">{{ $t("introduce.content") }}</div>
    </div>
    <div class="apply">
      <div class="apply-card" v-for="(item, index) in apply" :key="index">
        <img class="apply-card-img" :src="item.img" alt="应用图标" />
        <div class="apply-card-title">{{ $t(item.title) }}</div>
        <el-button
          class="apply-card-btn"
          type="warning"
          @click="handClick(item)"
          >{{ $t(item.button) }}</el-button
        >
      </div>
    </div>
  </div>
</template>
<script setup>
import spmvImg from "@/assets/images/spmv.png";
import spmmImg from "@/assets/images/spmm.png";
import graphImg from "@/assets/images/graph.png";

import { ElMessage } from "element-plus";

const { proxy } = getCurrentInstance();

const apply = reactive([
  {
    // spmv
    img: spmvImg,
    title: "apply.title1",
    button: "button.enter",
    path: "/spmv",
  },
  {
    // spmm
    img: spmmImg,
    title: "apply.title2",
    button: "button.enter",
    path: "",
  },
  {
    // graph
    img: graphImg,
    title: "apply.title3",
    button: "button.enter",
    path: "",
  },
]);

// 判断地址是否为网络地址
function isValidUrl(string) {
  const regex =
    /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
  return regex.test(string);
}

// 跳转
function handClick(row) {
  // 判断是否为移动端
  const ua = navigator.userAgent;
  let flag = /Android|webOS|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    ua,
  );
  if (row.path) {
    if (isValidUrl(row.path)) {
      window.open(row.path, flag ? "_self" : "_blank");
    } else {
      proxy.$router.push({ path: row.path });
    }
  } else {
    ElMessage({
      message: "功能建设中...",
      type: "warning",
    });
  }
}
</script>
<style lang="scss" scoped>
.home-container {
  .introduce {
    width: 100%;
    background: url("@/assets/images/introduce-bg.png") no-repeat center/cover;
    padding: 50px 10% 40px;
    .logo {
      height: 120px;
    }
    .title {
      margin-top: 34px;
      font-size: 24px;
      line-height: 32px;
    }
    .content {
      margin-top: 20px;
      font-size: 18px;
      line-height: 30px;
      white-space: pre-line;
    }
    .btn {
      margin-top: 20px;
    }
  }
  .apply {
    padding: 50px 10%;
    display: grid;
    // 定义每一列的列宽
    grid-template-columns: repeat(3, minmax(auto, 1fr));
    // 属性定义每一行的行高
    // grid-template-rows: 100px 100px 100px;
    // 设置行与行的间隔（行间距）
    grid-row-gap: 50px;
    // 设置列与列的间隔（列间距）
    grid-column-gap: 19px;
  }
}
.apply-card {
  // min-height: 375px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0 10% 30px;
  position: relative;
  text-align: center;
  &::after {
    content: "";
    width: 100%;
    height: calc(100% - 75px);
    background: linear-gradient(180deg, #e6f4ff 0%, #ffffff 100%);
    box-shadow: 0px 2px 5px 0px rgba(0, 0, 0, 0.15);
    border-radius: 4px;
    position: absolute;
    bottom: 0;
    left: 0;
    z-index: -1;
  }
  .apply-card-img {
    width: 150px;
    height: 150px;
  }
  .apply-card-title {
    margin-top: 20px;
    font-weight: bold;
    font-size: 24px;
    color: #333333;
    line-height: 32px;
  }
  .apply-card-btn {
    margin-top: 16px;
  }
}
@media only screen and (max-width: 750px) {
  .home-container {
    .introduce {
      padding: 20px 15% 20px;
      .logo {
        height: 30px;
      }
      .title {
        margin-top: 10px;
        font-size: 16px;
        line-height: 20px;
      }
      .content {
        margin-top: 10px;
        font-size: 14px;
        line-height: 18px;
      }
      .btn {
        margin-top: 10px;
      }
    }
    .apply {
      padding: 20px 10%;
      grid-template-columns: 1fr;
      grid-row-gap: 20px;
    }
  }
  .apply-card {
    padding: 0 10% 20px;
    &::after {
      height: calc(100% - 40px);
    }
    .apply-card-img {
      width: 80px;
      height: 80px;
    }
    .apply-card-title {
      margin-top: 10px;
      font-size: 20px;
      line-height: 25px;
    }
  }
}
</style>
