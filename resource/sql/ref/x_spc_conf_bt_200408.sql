CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `x_spc_conf_bt_200408`
--

DROP TABLE IF EXISTS `x_spc_conf_bt_200408`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `x_spc_conf_bt_200408` (
  `part_id` varchar(24) NOT NULL DEFAULT '*',
  `part_raw` varchar(16) DEFAULT NULL COMMENT '2020/04/08 lkchena',
  `route_id` varchar(24) NOT NULL DEFAULT '000' COMMENT '2020/04/08 lkchena',
  `mold_id` varchar(16) NOT NULL,
  `recipe_id` varchar(32) NOT NULL DEFAULT '*' COMMENT 'to be mold_id \n-- 2019/12/04 lkchena',
  `active` int(11) NOT NULL DEFAULT 1,
  `itype` int(11) NOT NULL DEFAULT 0 COMMENT 'default: 0 for spc\n-- 2019/12/04 lkchena: only zero for spc\n',
  `ope_no` varchar(7) NOT NULL COMMENT '2020/03/17 lkchena: for option flow, extend length to 7',
  `ope_name` varchar(32) DEFAULT 'MEAS_STEP',
  `stage_id` varchar(16) NOT NULL DEFAULT 'MEAS_STAGE',
  `stage_name` varchar(36) NOT NULL DEFAULT 'MEAS_STAGE',
  `stage_order` varchar(3) NOT NULL DEFAULT '100',
  `tool_type` varchar(12) NOT NULL DEFAULT '000' COMMENT '2020/04/08 lkchena',
  `tool_func` varchar(24) DEFAULT NULL COMMENT '2020/04/08 lkchena',
  `idxfield` int(11) NOT NULL,
  `field_name` varchar(24) NOT NULL,
  `field_desc` varchar(64) DEFAULT NULL,
  `iord` int(11) DEFAULT 1 COMMENT 'the filed order of report ',
  `csl` double DEFAULT NULL,
  `sign_mode` int(11) NOT NULL DEFAULT 0 COMMENT 'sign_mode:\n0: n/a  default\n1: ± x\n2: + x1 - x2\n3: ucl / lcl -- not implement\n\n2020/01/22 lkchena',
  `sign1` double DEFAULT NULL COMMENT 'sing1:  ± \nspec ex: ± 0.5 -->  csl - 0.5 < x <   csl - 0.5\n\nif zero or null, ignore it, check if sign2a,2b has setting\n\n2020/01/22 lkchena',
  `sign2a` double DEFAULT NULL COMMENT 'for ChinChih QC:\nspec:   csl - sign2b < x < csl + sing2a\n\nmust both has value, 2a is +, 2b - \n\n2020/01/22 lkchena',
  `sign2b` double DEFAULT NULL,
  `lsl` double DEFAULT NULL COMMENT '1. before save will calculating lsl,usl from sign1a,2a,2b  in UI\n2. lsl, usl will hide for calculating -- 2020/01/22 lkchena',
  `usl` double DEFAULT NULL COMMENT '1. before save will calculating lsl,usl from sign1a,2a,2b in UI\n2. lsl, usl will hide for calculating -- 2020/01/22 lkchena',
  `value_on` double DEFAULT NULL,
  `value_off` double DEFAULT NULL,
  `value_def` double DEFAULT NULL,
  `sample` int(11) DEFAULT 5 COMMENT 'rename to sample from sample_cnt 2019/12/17 lkchena\n2019/12/13 lkchena\nnot use xvalue1',
  `xvalue1` double DEFAULT 5 COMMENT 'to be sample count of spc, default 5\n-- 2019/12/04 lkchena',
  `xvalue2` double DEFAULT NULL,
  `xvalue3` double DEFAULT NULL,
  `decim` int(11) DEFAULT 0,
  `unit` varchar(8) DEFAULT NULL,
  `freq` int(11) NOT NULL DEFAULT -1 COMMENT 'data collect frequency: unit: second, -1: ignore(or event trigger)',
  `note` varchar(128) DEFAULT 'NULL',
  `rec_user` varchar(16) DEFAULT 'NULL',
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `x_spc_conf_bt_200408`
--

LOCK TABLES `x_spc_conf_bt_200408` WRITE;
/*!40000 ALTER TABLE `x_spc_conf_bt_200408` DISABLE KEYS */;
INSERT INTO `x_spc_conf_bt_200408` VALUES ('AAFFS1',NULL,'000','AAFFS1','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','FORMING_STEP','100','910','量測-高度1',1,'Height_t','總高度',1001,20.5,1,0.45,NULL,NULL,20.05,20.95,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,0,'NULL','SYS','2020/04/08 11:45:56',NULL,'2020-04-08 03:45:56.462151'),('AAFFS1',NULL,'000','AAFFS1','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','FORMING_STEP','100','930','量測-重量',2,'Weight','重量',1002,11,2,NULL,0.8,0.5,10.5,11.8,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,0,'NULL','SYS','2020/04/08 11:46:12',NULL,'2020-04-08 03:46:12.694788'),('AAGGEE',NULL,'000','AAGGEE','*',0,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,1,'Weight','重量3-TEST2',1,22,0,NULL,NULL,NULL,21,23,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,-1,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('ACQ51F-1',NULL,'000','ACQ51F-1','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,1,'Height_t','總高度',1,30.5,0,NULL,NULL,NULL,30,31,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,0,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('ACQ51F-1',NULL,'000','ACQ51F-1','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,2,'Weight','重量',2,14,0,NULL,NULL,NULL,13,15,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,0,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('BBCES03',NULL,'000','BBCES03','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,1,'Height_t','總高度',1,40,0,NULL,NULL,NULL,39,41,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,-1,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('BBCES03',NULL,'000','BBCES03','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,2,'Weight','重量',2,30,0,NULL,NULL,NULL,29,31,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,-1,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('SHC-366',NULL,'000','SHC-366','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,1,'Weight','重量',1,20,0,NULL,NULL,NULL,19,21,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,-1,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('SHC-366',NULL,'000','SHC-366','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,2,'Height_t','高度',1,30,1,0.5,NULL,NULL,29.5,30.5,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,-1,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664'),('SHC-366',NULL,'000','SHC-366','*',1,0,'100.300','MEAS_STEP','MEAS_STAGE','MEAS_STAGE','100','000',NULL,3,'Hardness','硬度',1,80,2,NULL,10,5,75,90,NULL,NULL,NULL,5,5,NULL,NULL,0,NULL,-1,'NULL','SYS','2020/01/23 15:14:29',NULL,'2020-03-17 02:22:59.519664');
/*!40000 ALTER TABLE `x_spc_conf_bt_200408` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:20
