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
-- Table structure for table `dm_step_bt`
--

DROP TABLE IF EXISTS `dm_step_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dm_step_bt` (
  `area_id` varchar(12) NOT NULL,
  `stage_id` varchar(16) NOT NULL,
  `ope_code` varchar(3) NOT NULL COMMENT '改成 3碼(資料庫全部ope_no改成7碼) -- 2020/03/17 lkchena\nope_no 的尾兩碼(main step, not sub steps): 不會超過100個吧!...超過的話直接增加外送 "stage"... ',
  `ope_name` varchar(36) DEFAULT NULL COMMENT 'step description',
  `ws_type` varchar(12) NOT NULL DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `tool_type1` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `tool_type2` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `tool_type3` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `in_out` int(11) NOT NULL DEFAULT 0 COMMENT 'default: 0: 廠內  1: 外注 1x: maybe A,B factory\n2020/03/25 lkchena',
  `rec_user` varchar(16) DEFAULT NULL COMMENT 'update user: log NT account',
  `rec_time` varchar(19) DEFAULT NULL COMMENT 'record time',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`stage_id`,`ope_code`,`ws_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dm_step_bt`
--

LOCK TABLES `dm_step_bt` WRITE;
/*!40000 ALTER TABLE `dm_step_bt` DISABLE KEYS */;
INSERT INTO `dm_step_bt` VALUES ('S001','Sintering','050','Sorter - 換棧板','20','85','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.139085'),('S001','Sintering','100','燒結-主製程','20','20','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.140089'),('S001','Sintering','105','燒結-爐尾','21','20','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.140089'),('S001','Sintering','200','燒結防鏽','20','25','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.140089'),('S001','Sintering','200','燒結防鏽','21','25','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.140089'),('S001','Sintering','210','燒結乾振','20','30','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.140089'),('S001','Sintering','210','燒結乾振','21','30','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.140089'),('S001','Sintering','220','燒結乾振防鏽','20','25','30','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','220','燒結乾振防鏽','21','25','30','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','230','燒結滲銅防鏽','20','40','25','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','230','燒結滲銅防鏽','21','40','25','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','240','燒結焊接','20','50','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','240','燒結焊接','21','50','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','250','燒結除油','20','60','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','250','燒結除油','21','60','0','0',0,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','705','外注-回火','21','71','0','0',1,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088'),('S001','Sintering','710','外注-測試廠商','21','72','0','0',1,'SYS','2020/03/17 14:31:25',NULL,'2020-03-21 09:09:46.141088');
/*!40000 ALTER TABLE `dm_step_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:24
