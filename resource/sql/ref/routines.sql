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
-- Temporary view structure for view `daul`
--

DROP TABLE IF EXISTS `daul`;
/*!50001 DROP VIEW IF EXISTS `daul`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE VIEW `daul` AS SELECT 
 1 AS `DUMMY`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `dm_flow_bv`
--

DROP TABLE IF EXISTS `dm_flow_bv`;
/*!50001 DROP VIEW IF EXISTS `dm_flow_bv`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE VIEW `dm_flow_bv` AS SELECT 
 1 AS `part_id`,
 1 AS `part_raw`,
 1 AS `route_id`,
 1 AS `rd_flag`,
 1 AS `mold_id`,
 1 AS `powder_type`,
 1 AS `lot_size_spec`,
 1 AS `box_size_spec`,
 1 AS `wafer_size_spec`,
 1 AS `cust_id`,
 1 AS `cust_name`,
 1 AS `ope_no`,
 1 AS `ope_name`,
 1 AS `stage_id`,
 1 AS `stage_name`,
 1 AS `stage_order`,
 1 AS `tool_grp_id`,
 1 AS `tool_grp`,
 1 AS `ws_type`,
 1 AS `tool_type1`,
 1 AS `tool_type2`,
 1 AS `tool_type3`,
 1 AS `in_out`,
 1 AS `area_id`,
 1 AS `area_name`,
 1 AS `proc_time`,
 1 AS `weight_above`,
 1 AS `weight_below`,
 1 AS `density_above`,
 1 AS `density_below`,
 1 AS `hardness`,
 1 AS `material`,
 1 AS `layout_doc`,
 1 AS `unit_label`,
 1 AS `unit_type`,
 1 AS `note`*/;
SET character_set_client = @saved_cs_client;

--
-- Temporary view structure for view `temp_dcop_drcass1_v`
--

DROP TABLE IF EXISTS `temp_dcop_drcass1_v`;
/*!50001 DROP VIEW IF EXISTS `temp_dcop_drcass1_v`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE VIEW `temp_dcop_drcass1_v` AS SELECT 
 1 AS `VsPrimaryKey`,
 1 AS `CumQty`*/;
SET character_set_client = @saved_cs_client;

--
-- Final view structure for view `daul`
--

/*!50001 DROP VIEW IF EXISTS `daul`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8 */;
/*!50001 SET character_set_results     = utf8 */;
/*!50001 SET collation_connection      = utf8_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`%` SQL SECURITY DEFINER */
/*!50001 VIEW `daul` AS select 'X' AS `DUMMY` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `dm_flow_bv`
--

/*!50001 DROP VIEW IF EXISTS `dm_flow_bv`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8 */;
/*!50001 SET character_set_results     = utf8 */;
/*!50001 SET collation_connection      = utf8_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `dm_flow_bv` AS select `a`.`part_id` AS `part_id`,`a`.`part_raw` AS `part_raw`,`a`.`route_id` AS `route_id`,`a`.`rd_flag` AS `rd_flag`,`a`.`mold_id` AS `mold_id`,`a`.`powder_type` AS `powder_type`,`a`.`lot_size_spec` AS `lot_size_spec`,`a`.`box_size_spec` AS `box_size_spec`,`a`.`wafer_size_spec` AS `wafer_size_spec`,`a`.`cust_id` AS `cust_id`,`a`.`cust_name` AS `cust_name`,`b`.`ope_no` AS `ope_no`,`b`.`ope_name` AS `ope_name`,`b`.`stage_id` AS `stage_id`,`b`.`stage_name` AS `stage_name`,`b`.`stage_order` AS `stage_order`,`b`.`tool_grp_id` AS `tool_grp_id`,`b`.`tool_grp` AS `tool_grp`,`b`.`ws_type` AS `ws_type`,`b`.`tool_type1` AS `tool_type1`,`b`.`tool_type2` AS `tool_type2`,`b`.`tool_type3` AS `tool_type3`,`b`.`in_out` AS `in_out`,`b`.`area_id` AS `area_id`,`b`.`area_name` AS `area_name`,`b`.`proc_time` AS `proc_time`,`a`.`weight_above` AS `weight_above`,`a`.`weight_below` AS `weight_below`,`a`.`density_above` AS `density_above`,`a`.`density_below` AS `density_below`,`a`.`hardness` AS `hardness`,`a`.`material` AS `material`,`a`.`layout_doc` AS `layout_doc`,`a`.`unit_label` AS `unit_label`,`a`.`unit_type` AS `unit_type`,`a`.`note` AS `note` from (`dm_part_bt` `a` join `dm_route_bt` `b` on(`a`.`route_id` = `b`.`route_id`)) order by `b`.`ope_no` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Final view structure for view `temp_dcop_drcass1_v`
--

/*!50001 DROP VIEW IF EXISTS `temp_dcop_drcass1_v`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8 */;
/*!50001 SET character_set_results     = utf8 */;
/*!50001 SET collation_connection      = utf8_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `temp_dcop_drcass1_v` AS select `a`.`VsPrimaryKey` AS `VsPrimaryKey`,`a`.`n16` AS `CumQty` from `dcop_collect_bt` `a` where 1 = 1 and `a`.`tool_id` = 'AF15T01' and `a`.`time_order` >= '2019/11/23 09:00:00' and `a`.`time_order` <= '2019/11/23 09:59:59' and `a`.`VsPrimaryKey` like '%_SS' order by `a`.`time_order` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;

--
-- Dumping events for database 'mes_omi'
--
/*!50106 SET @save_time_zone= @@TIME_ZONE */ ;
/*!50106 DROP EVENT IF EXISTS `job_check_csv_data` */;
DELIMITER ;;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;;
/*!50003 SET character_set_client  = utf8mb4 */ ;;
/*!50003 SET character_set_results = utf8mb4 */ ;;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;;
/*!50003 SET @saved_time_zone      = @@time_zone */ ;;
/*!50003 SET time_zone             = 'SYSTEM' */ ;;
/*!50106 CREATE*/ /*!50117 DEFINER=`root`@`localhost`*/ /*!50106 EVENT `job_check_csv_data` ON SCHEDULE EVERY 8 HOUR STARTS '2020-02-20 16:15:00' ON COMPLETION PRESERVE DISABLE COMMENT 'Confirm that the CSV data is complete.' DO BEGIN
DECLARE iCsvCount int DEFAULT 0;

SET iCsvCount = 
(
 SELECT COUNT(*) FROM dcop_event_bt
 WHERE claim_time >= date_sub(now(), interval  9 HOUR)
   AND claim_time <  date_sub(now(), interval  1 HOUR)
   AND kind = 0
   AND field_value = 9
);

INSERT INTO sys_job_log_bt (Job_Name, time_start, job_status, note) VALUES ('job_check_csv_data', NOW(), 9, iCsvCount); 


																			
END */ ;;
/*!50003 SET time_zone             = @saved_time_zone */ ;;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;;
/*!50003 SET character_set_client  = @saved_cs_client */ ;;
/*!50003 SET character_set_results = @saved_cs_results */ ;;
/*!50003 SET collation_connection  = @saved_col_connection */ ;;
/*!50106 DROP EVENT IF EXISTS `job_day_event` */;;
DELIMITER ;;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;;
/*!50003 SET character_set_client  = utf8mb4 */ ;;
/*!50003 SET character_set_results = utf8mb4 */ ;;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;;
/*!50003 SET @saved_time_zone      = @@time_zone */ ;;
/*!50003 SET time_zone             = 'SYSTEM' */ ;;
/*!50106 CREATE*/ /*!50117 DEFINER=`root`@`localhost`*/ /*!50106 EVENT `job_day_event` ON SCHEDULE EVERY 1 DAY STARTS '2020-02-27 01:15:00' ON COMPLETION PRESERVE DISABLE DO BEGIN
   
   
  
  
 declare  stJob_Name varchar(32) DEFAULT 'job_day_event';
 declare  iElp_Spec int DEFAULT 30;
 declare  dExec_Time datetime DEFAULT NOW();  
 declare  stErrorLog varchar(64);
 declare  iResult_1_ok_0_ng INT;    
  
 declare  iDcop_keep_day INT;
 declare  iDcop_Hist_day INT;
 declare  iDcop_ct_keep_day INT; 
 declare  iDcop_Hist_ct_day INT; 
 
 declare  iLog_keep_day INT;
 declare  iLog_Hist_day INT;  
 
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
																				  
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              
                                                              select '000' INTO stErrorLog; 
  
  SELECT a.ivalue1, a.ivalue2, a.ivalue3, a.ivalue4 INTO iDcop_keep_day, iDcop_Hist_day, iDcop_ct_keep_day, iDcop_Hist_ct_day
  FROM sys_param_conf_bt a
  WHERE 1=1
  and a.param_id = 'DCOP-001';                                      													      
																													                                                                                                                                         select '001' INTO stErrorLog; 
																													                                                                             
  SELECT a.ivalue1, a.ivalue2 INTO iLog_keep_day, iLog_Hist_day
  FROM sys_param_conf_bt a
  WHERE 1=1
  and a.param_id = 'LOG-001';                                                                       													                                                                                                                       													                                                                                                                															  
                                                              select '002' INTO stErrorLog;
                                                              
  
  call dcop_move_hist_sp(iDcop_keep_day, iDcop_ct_keep_day);
                                                              select '003' INTO stErrorLog;
                                                              
  
  call dcop_hist_del_sp(iDcop_Hist_day, iDcop_Hist_ct_day);                                                            
                                                              select '004' INTO stErrorLog;
                                                              
  
  call log_move_hist_sp(iLog_keep_day);                                                            
                                                              select '005' INTO stErrorLog;
                                                              
  
  call log_hist_del_sp(iLog_Hist_day);                                                             
                                                              select '900' INTO stErrorLog;
  
  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 9,stJob_Name,dExec_Time,iElp_Spec,'');
																					
  
END */ ;;
/*!50003 SET time_zone             = @saved_time_zone */ ;;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;;
/*!50003 SET character_set_client  = @saved_cs_client */ ;;
/*!50003 SET character_set_results = @saved_cs_results */ ;;
/*!50003 SET collation_connection  = @saved_col_connection */ ;;
DELIMITER ;
/*!50106 SET TIME_ZONE= @save_time_zone */ ;

--
-- Dumping routines for database 'mes_omi'
--
/*!50003 DROP FUNCTION IF EXISTS `fn_cr8_box_id` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` FUNCTION `fn_cr8_box_id`(
 stLOT_LEAD varchar(64), iCUM int, iUnit_Type int
) RETURNS varchar(64) CHARSET utf8
BEGIN
-- tip1: please see fn_cr8_box_id to get detail -- 2020/05/02 lkchena
-- rule: .01~99, A0~A9, ..., z0~z9. total: 619: 99+26*10*2
--       .00 resereved

-- 2020/04/29 lkchena: create new lot_id
/*
 select fn_cr8_box_id('A01000337',0,0);
 select fn_cr8_box_id('A01000337',99,0);
 select fn_cr8_box_id('A01000337',100,0);
 select fn_cr8_box_id('A01000337',100,1); -- by weight
 select fn_cr8_box_id('A01000337',109,0);
 select fn_cr8_box_id('A01000337',110,0);
 select fn_cr8_box_id('A01000337',619,0); 
 */

DECLARE stWafer_Id varchar(64) DEFAULT null;
DECLARE stSuffix   varchar(64) DEFAULT null;


DECLARE iCUM2 int DEFAULT 0;
DECLARE iDiv      int DEFAULT 0; -- 商數
DECLARE iMod      int DEFAULT 0; -- 餘數

  -- 0.0 zero
  set stWafer_Id = '';

  -- 0.1 judge unit_type: 0: by piece 1: by weight
  if iUnit_Type = 1 then
    RETURN (stWafer_Id);
  end if;

  -- 0.2 zero
  set iCUM2 = iCUM;

  -- 1.0 normal case
  if iCUM2 < 100 then
    
    SET stSuffix   = LPAD(iCUM2,2,'0');    
    set stWafer_Id = concat(stLOT_LEAD,'.',stSuffix);
        
  else

    -- 1.2 parse : 
    -- select char(65),char(90),char(97),char(122) -- A,Z,a,z
    
	set    iCUM2 =   iCUM - 100;   
    select iCUM2 div 10 into iDiv;
    select iCUM2 mod 10 into iMod;
    
    -- 1.3 suffix A~Z, a~z
    if iDiv < 26 then
      SET stSuffix   = concat(CHAR(65+iDiv),LPAD(iMod,1,'0'));-- A ~ Z
    else
      set iDiv = iDiv - 26;
      SET stSuffix   = concat(CHAR(97+iDiv),LPAD(iMod,1,'0'));-- a ~ z    
    end if;
    
    -- 1.5 result
    set stWafer_Id = concat(stLOT_LEAD,'.',stSuffix); -- concat(iCUM2,' - ', iDiv,' - ', iMod); 
  
  end if;
  
RETURN (stWafer_Id);
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP FUNCTION IF EXISTS `fn_cr8_dum_box_no` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` FUNCTION `fn_cr8_dum_box_no`(
 stLOT_HEADER varchar(64), iLOT_CUM int, iLOT_LEN int
) RETURNS varchar(64) CHARSET utf8
BEGIN
-- 2020/05/02 lkchena: create dummy box_id
-- tip1: update cumlated value of sys_param_bt table in outside block, not here, saving time
-- tip2: use stLot_Id replace box_id to save code
/*
 select fn_cr8_dum_box_id('Z',100,5);-- as lot_id;
 select fn_cr8_dum_box_id('Z',101,5);-- as lot_id;
 select fn_cr8_dum_box_id('Z',7000000,5);-- as lot_id;
 select fn_cr8_dum_box_id('Z',99999,5);-- as lot_id;
 select fn_cr8_dum_box_id('Z',100000,5);-- as lot_id;
*/

DECLARE stLot_Id  varchar(64) DEFAULT null;
DECLARE iLOT_CUM2 int DEFAULT 0;
DECLARE iMax      int DEFAULT 0;

  -- 1.0 get iMax
  select CASE iLOT_LEN  
		  WHEN 5 THEN 99999
          WHEN 6 THEN 999999 
          WHEN 7 THEN 9999999
          WHEN 8 THEN 99999999
          WHEN 9 THEN 999999999
          WHEN 4 THEN 9999
          WHEN 3 THEN 999
          WHEN 2 THEN 99
          ELSE 10000  
       END  into iMax;

  -- 1.2 deal iLot_Cum
   set iLOT_CUM2 = iLOT_CUM;
  
   WHILE ( iLOT_CUM2 > iMax)  DO   
    set iLOT_CUM2 =  iLOT_CUM2 - iMax;
   END WHILE;                                       

  -- 2. get lot_id  
  set stLot_Id =  CONCAT(stLOT_HEADER , LPAD(iLOT_CUM2,iLOT_LEN, '0'));
  -- select stLot_Id;
  
RETURN (stLot_Id);
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP FUNCTION IF EXISTS `fn_cr8_lot_id` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` FUNCTION `fn_cr8_lot_id`(
 stLOT_HEADER varchar(64), stNODE_HEADER varchar(64), iLOT_CUM int, iLOT_LEN int
) RETURNS varchar(64) CHARSET utf8
BEGIN
-- 2020/04/29 lkchena: create new lot_id
-- select fn_cr8_lot_id('A','0R',100,5);-- as lot_id;
-- select fn_cr8_lot_id('A','0R',7000000,5);-- as lot_id;
-- select fn_cr8_lot_id('A','0R',99999,5);-- as lot_id;
-- select fn_cr8_lot_id('A','0R',100000,5);-- as lot_id;
-- org: select   CONCAT('A','01',LPAD(7000000,5, '0'),'.0' ) as lot_id

DECLARE stLot_Id  varchar(64) DEFAULT null;
DECLARE iLOT_CUM2 int DEFAULT 0;
DECLARE iMax      int DEFAULT 0;

  -- 1.0 get iMax
  select CASE iLOT_LEN  
		  WHEN 5 THEN 99999
          WHEN 6 THEN 999999 
          WHEN 7 THEN 9999999
          WHEN 8 THEN 99999999
          WHEN 9 THEN 999999999
          WHEN 4 THEN 9999
          WHEN 3 THEN 999
          WHEN 2 THEN 99
          ELSE 10000  
       END  into iMax;
  
  -- 1.2 deal iLot_Cum
   set iLOT_CUM2 = iLOT_CUM;
  
   WHILE ( iLOT_CUM2 > iMax)  DO   
    set iLOT_CUM2 =  iLOT_CUM2 - iMax;
   END WHILE;                                       

  -- 2. get lot_id  
  set stLot_Id =  CONCAT(stLOT_HEADER , stNODE_HEADER,LPAD(iLOT_CUM2,iLOT_LEN, '0'),'.0' );
                   -- concat(stLOT_HEADER , stNODE_HEADER );
  
  -- select stLot_Id;
  
RETURN (stLot_Id);
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP FUNCTION IF EXISTS `fn_cr8_wafer_id` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` FUNCTION `fn_cr8_wafer_id`(
 stLOT_LEAD varchar(64), iCUM int, iUnit_Type int
) RETURNS varchar(64) CHARSET utf8
BEGIN
-- tip: stLOT_LEAD: w/o Suffix, ex: A12345678.00 --> A12345678 <-- let function run fast
-- wafer .000 rule:
-- A. 正常狀況: 001~999, A00~A99, B00~B99, ... ,Z00~Z99  -- total: 3599 (999+26*100)
--      add: a~z, total: 6199 (999+26*100*2) 
-- B. 秤重小工件: 應該不能計到 wafer_id, 應該直接用 lot_size 與重量, 不用解析 wafer_id - "待確認"
-- //tip: wafer .000 不夠用, 因為棧板, 最少 1000, 最多 10000(小工件,無法計數,用秤重的),要動用到英文

-- 2020/04/29 lkchena: create new lot_id
/*
 select fn_cr8_wafer_id('A01000337',0,0);
 select fn_cr8_wafer_id('A01000337',999,0);
 select fn_cr8_wafer_id('A01000337',999,1); -- by weight
 select fn_cr8_wafer_id('A01000337',1000,0);
 select fn_cr8_wafer_id('A01000337',2000,0);
 select fn_cr8_wafer_id('A01000337',3000,0);
 select fn_cr8_wafer_id('A01000337',3599,0);
 select fn_cr8_wafer_id('A01000337',3600,0);
 select fn_cr8_wafer_id('A01000337',3601,0);
 select fn_cr8_wafer_id('A01000337',6199,0);
 select fn_cr8_wafer_id('A01000337',6200,0);
 */

DECLARE stWafer_Id varchar(64) DEFAULT null;
DECLARE stSuffix   varchar(64) DEFAULT null;


DECLARE iCUM2 int DEFAULT 0;
DECLARE iDiv      int DEFAULT 0; -- 商數
DECLARE iMod      int DEFAULT 0; -- 餘數

  -- 0.0 zero
  set stWafer_Id = '';

  -- 0.1 judge unit_type: 0: by piece 1: by weight
  if iUnit_Type = 1 then
    RETURN (stWafer_Id);
  end if;

  -- 0.2 zero
  set iCUM2 = iCUM;

  -- 1.0 normal case
  if iCUM2 < 1000 then
    
    SET stSuffix   = LPAD(iCUM2,3,'0');    
    set stWafer_Id = concat(stLOT_LEAD,'.',stSuffix);
        
  else

    -- 1.2 parse : 
    -- select char(65),char(90),char(97),char(122) -- A,Z,a,z
    
	set    iCUM2 =   iCUM - 1000;   
    select iCUM2 div 100 into iDiv;
    select iCUM2 mod 100 into iMod;
    
    -- 1.3 suffix A~Z, a~z
    if iDiv < 26 then
      SET stSuffix   = concat(CHAR(65+iDiv),LPAD(iMod,2,'0'));-- A ~ Z
    else
      set iDiv = iDiv - 26;
      SET stSuffix   = concat(CHAR(97+iDiv),LPAD(iMod,2,'0'));-- a ~ z    
    end if;
    
    -- 1.5 result
    set stWafer_Id = concat(stLOT_LEAD,'.',stSuffix); -- concat(iCUM2,' - ', iDiv,' - ', iMod); 
  
  end if;
  
RETURN (stWafer_Id);
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP FUNCTION IF EXISTS `get_shift` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` FUNCTION `get_shift`(Date1 DATETIME, iShift INT) RETURNS datetime
    COMMENT 'date1: default system date   iShift: 1,2,3:( First Shift, Second Shift, Third Shift.)    11,12: A/B ( day shift 和 night shift. )'
BEGIN





  DECLARE resDate DATETIME DEFAULT NOW();
  DECLARE stDate varchar(10);
     
  SELECT DATE_FORMAT(DATE(date1),'%Y/%m/%d') INTO stDate;

  select 
    CASE iShift 
         WHEN  1 THEN STR_TO_DATE( concat(stDate,' ','07:00:00'), '%Y/%m/%d %T')  
         WHEN  2 THEN STR_TO_DATE( concat(stDate,' ','15:00:00'), '%Y/%m/%d %T')
         WHEN  3 THEN STR_TO_DATE( concat(stDate,' ','23:00:00'), '%Y/%m/%d %T')
         WHEN 11 THEN STR_TO_DATE( concat(stDate,' ','07:20:00'), '%Y/%m/%d %T')
         WHEN 12 THEN STR_TO_DATE( concat(stDate,' ','19:20:00'), '%Y/%m/%d %T')
     ELSE             STR_TO_DATE( concat(stDate,' ','07:00:00'), '%Y/%m/%d %T') END into resDate;

RETURN resDate;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP FUNCTION IF EXISTS `hello` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` FUNCTION `hello`() RETURNS int(11)
BEGIN

RETURN 9;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP FUNCTION IF EXISTS `ump_get_lot_id_fn` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` FUNCTION `ump_get_lot_id_fn`(
	`iType` INT
) RETURNS varchar(12) CHARSET utf8
    COMMENT 'iType: 0: current lot_id, 1: get next lot_it'
BEGIN



  declare stResLot_ID varchar(12) DEFAULT 'A0000000.00'; 
  
  DECLARE stHeader varchar(2) DEFAULT 'A';
  DECLARE iCum    int DEFAULT 1;
  DECLARE iLen     int DEFAULT 7;

  
  IF iType = 1 THEN 
    
    update sys_param_conf_bt  set ivalue1 = ivalue1 + 1
    where param_id = 'UMP-L002'; 

  END IF;
      
  
   SELECT value1  INTO stHeader from sys_param_conf_bt where param_id = 'UMP-L001'; 
   SELECT ivalue1 INTO iCum     from sys_param_conf_bt where param_id = 'UMP-L002'; 
   SELECT ivalue1 INTO iLen     from sys_param_conf_bt where param_id = 'UMP-L003'; 
 
  
    SELECT concat(stHeader,LPAD(iCum, iLen, '0'),'.00') INTO stResLot_ID;

RETURN stResLot_ID;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `dcop_hist_del_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `dcop_hist_del_sp`(
	IN `iDelHist_day` INT,
	IN `iDelHist_ct_day` INT
)
    COMMENT 'dcop history delete  exceed day data'
BEGIN
 declare  stJob_Name varchar(32) DEFAULT 'dcop_hist_del_sp';
 declare  iElp_Spec int DEFAULT 30;
 declare  dExec_Time datetime DEFAULT NOW();  
 declare  stErrorLog varchar(64);
 declare  iResult_1_ok_0_ng INT;   

 
     
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
																				  
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              
                                                              select '000' INTO stErrorLog; 

  
  DELETE FROM dcop_collect_bth 
  where claim_time <= DATE_SUB(NOW(),INTERVAL iDelHist_day DAY);   
  
                                                              select '001' INTO stErrorLog;
  
  DELETE FROM dcop_collect_cth                                                                 													                                                                                                                													
  where claim_time <= DATE_SUB(NOW(),INTERVAL iDelHist_ct_day DAY);
  
                                                              
                                                              select '900' INTO stErrorLog;
  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');
																					


  
  

  

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `dcop_move_hist_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `dcop_move_hist_sp`(
	IN `iDcop_keep_day` INT,
	IN `iDcop_keep_ct_day` INT
)
    COMMENT 'only keep ??? days, then move to history table'
BEGIN
 declare  stJob_Name varchar(32) DEFAULT 'dcop_move_hist_sp';
 declare  iElp_Spec int DEFAULT 30;
 declare  dExec_Time datetime DEFAULT NOW();  
 declare  stErrorLog varchar(64);
 declare  iResult_1_ok_0_ng INT;   


 
     
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
																				  
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              
                                                              select '000' INTO stErrorLog; 

                                                                         													                                                    
  INSERT dcop_collect_bth 
  SELECT * FROM dcop_collect_bt a 
  where 1=1 and a.claim_time <= DATE_SUB(NOW(),INTERVAL iDcop_keep_day DAY);                                                                      													                                                                                                                															  
                                                              select '001' INTO stErrorLog;
  
                                                              
  INSERT dcop_collect_cth 
  SELECT * FROM dcop_collect_ct a 
  where 1=1 and a.claim_time <= DATE_SUB(NOW(),INTERVAL iDcop_keep_ct_day DAY);                                                                      													                                                                                                                															  
                                                              select '002' INTO stErrorLog;  
  
                                                              
 
  DELETE FROM dcop_collect_bt 
  where claim_time <= DATE_SUB(NOW(),INTERVAL iDcop_keep_day DAY);
  
                                                              SELECT '003' INTO stErrorLog;


  DELETE FROM dcop_collect_ct 
  where claim_time <= DATE_SUB(NOW(),INTERVAL iDcop_keep_ct_day DAY);
  
                                                              select '900' INTO stErrorLog;  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');
																					


  
  

  

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `dcop_raw_capt_s1_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `dcop_raw_capt_s1_sp`(IN stDate_Re varchar(10), stTool_Id varchar(7),stLast1 varchar(19),stLast2 varchar(19), stKeyField varchar(7), iCumQTY int)
BEGIN
DECLARE cr_stack_depth INTEGER DEFAULT cr_debug.ENTER_MODULE2('dcop_raw_capt_s1_sp', 'mes', 7, 100632) ; 

   

  
  
  
  
  
  
  
  
  
  

  DECLARE stJob_Name varchar(32) DEFAULT "dcop_raw_capt_s1_sp";
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE stVsPrimaryKey varchar(48);
  DECLARE iCumQTY_Start int DEFAULT 0;
  DECLARE iCumQTY_Raw int DEFAULT 0;
  DECLARE stSQL_SS_RE_Flag varchar(64);
  

  DECLARE doneCursor int DEFAULT 0;
  DECLARE curMain CURSOR FOR SELECT VsPrimaryKey, CumQty from temp_dcop_drcass1_v;
    
  DECLARE CONTINUE HANDLER FOR NOT FOUND BEGIN
DECLARE cr_stack_depth INTEGER DEFAULT cr_debug.ENTER_HANDLER('dcop_raw_capt_s1_sp_Handler', 'dcop_raw_capt_s1_sp', 'mes', 7, 100632) ;
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iTmp', iTmp, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCNT', iCNT, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stVsPrimaryKey', stVsPrimaryKey, 'varchar(48)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Start', iCumQTY_Start, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Raw', iCumQTY_Raw, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stSQL_SS_RE_Flag', stSQL_SS_RE_Flag, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('doneCursor', doneCursor, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stDate_Re', stDate_Re, 'varchar(10)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stTool_Id', stTool_Id, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast1', stLast1, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast2', stLast2, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stKeyField', stKeyField, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY', iCumQTY, 'int', cr_stack_depth) ;
CALL cr_debug.TRACE(43, 43, 41, 60, cr_stack_depth) ;
SET doneCursor = 1;
CALL cr_debug.UPDATE_WATCH3('doneCursor', doneCursor, '', cr_stack_depth) ;
CALL cr_debug.LEAVE_MODULE(cr_stack_depth - 1) ;
 END;

                                                     DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                                       BEGIN
DECLARE cr_stack_depth INTEGER DEFAULT cr_debug.ENTER_HANDLER('dcop_raw_capt_s1_sp_Handler', 'dcop_raw_capt_s1_sp', 'mes', 7, 100632) ;
                                                         CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iTmp', iTmp, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCNT', iCNT, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stVsPrimaryKey', stVsPrimaryKey, 'varchar(48)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Start', iCumQTY_Start, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Raw', iCumQTY_Raw, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stSQL_SS_RE_Flag', stSQL_SS_RE_Flag, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('doneCursor', doneCursor, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stDate_Re', stDate_Re, 'varchar(10)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stTool_Id', stTool_Id, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast1', stLast1, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast2', stLast2, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stKeyField', stKeyField, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY', iCumQTY, 'int', cr_stack_depth) ;
CALL cr_debug.TRACE(46, 46, 55, 60, cr_stack_depth) ;
CALL cr_debug.TRACE(47, 47, 57, 89, cr_stack_depth) ;
SELECT 0 INTO iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, '', cr_stack_depth) ;
                                                         CALL cr_debug.TRACE(48, 48, 57, 66, cr_stack_depth) ;
ROLLBACK;
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iTmp', iTmp, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCNT', iCNT, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stVsPrimaryKey', stVsPrimaryKey, 'varchar(48)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Start', iCumQTY_Start, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Raw', iCumQTY_Raw, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stSQL_SS_RE_Flag', stSQL_SS_RE_Flag, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('doneCursor', doneCursor, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stDate_Re', stDate_Re, 'varchar(10)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stTool_Id', stTool_Id, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast1', stLast1, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast2', stLast2, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stKeyField', stKeyField, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY', iCumQTY, 'int', cr_stack_depth) ;
                                                         CALL cr_debug.TRACE(49, 49, 57, 129, cr_stack_depth) ;
CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog);
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ; 
                                                         CALL cr_debug.TRACE(50, 50, 57, 82, cr_stack_depth) ;
SELECT iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
                                                       CALL cr_debug.TRACE(51, 51, 55, 58, cr_stack_depth) ;
CALL cr_debug.LEAVE_MODULE(cr_stack_depth - 1) ;
END;                                        
                                                       CALL cr_debug.UPDATE_WATCH3('stDate_Re', stDate_Re, 'varchar(10)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stTool_Id', stTool_Id, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast1', stLast1, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLast2', stLast2, 'varchar(19)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stKeyField', stKeyField, 'varchar(7)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY', iCumQTY, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iTmp', iTmp, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCNT', iCNT, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stVsPrimaryKey', stVsPrimaryKey, 'varchar(48)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Start', iCumQTY_Start, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Raw', iCumQTY_Raw, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stSQL_SS_RE_Flag', stSQL_SS_RE_Flag, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('doneCursor', doneCursor, 'int', cr_stack_depth) ;
CALL cr_debug.TRACE(2, 2, 0, 5, cr_stack_depth) ;
CALL cr_debug.TRACE(52, 52, 55, 118, cr_stack_depth) ;
CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, '', cr_stack_depth) ;
                                                       CALL cr_debug.TRACE(53, 53, 55, 78, cr_stack_depth) ;
SET stErrorLog = '000';
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
  
  
  
 
  
   CALL cr_debug.TRACE(59, 59, 3, 59, cr_stack_depth) ;
set stSQL_SS_RE_Flag = " and VsPrimaryKey LIKE '%_SS' ";
CALL cr_debug.UPDATE_WATCH3('stSQL_SS_RE_Flag', stSQL_SS_RE_Flag, '', cr_stack_depth) ;

   CALL cr_debug.TRACE(61, 61, 3, 138, cr_stack_depth) ;
SELECT ROUND(time_to_sec(TIMEDIFF(STR_TO_DATE(stLast2, "%Y/%m/%d %H:%i:%S"), STR_TO_DATE(stLast1, "%Y/%m/%d %H:%i:%S")))/60) INTO iTmp;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('iTmp', iTmp, '', cr_stack_depth) ;

   
   
   CALL cr_debug.TRACE(65, 67, 3, 10, cr_stack_depth) ;
IF iTmp > 60 THEN
     CALL cr_debug.TRACE(66, 66, 5, 61, cr_stack_depth) ;
set stSQL_SS_RE_Flag = " and VsPrimaryKey LIKE '%_RE' ";
CALL cr_debug.UPDATE_WATCH3('stSQL_SS_RE_Flag', stSQL_SS_RE_Flag, '', cr_stack_depth) ;
   END IF;   

  
                              CALL cr_debug.TRACE(70, 78, 30, 75, cr_stack_depth) ;
SET @v = concat('CREATE OR REPLACE VIEW temp_dcop_drcass1_v ',
                                                ' as SELECT a.VsPrimaryKey, ', stKeyField,' as CumQty ', 
                                                '   FROM dcop_collect_bt a ',
                                                ' where 1=1 ',
                                                '   and a.tool_id =  ''',stTool_Id, ''' ', 
                                                '   and a.time_order >= ''',stLast1, ''' ', 
                                                '   and a.time_order <= ''',stLast2, ''' ', 
                                                stSQL_SS_RE_Flag, 
                                                ' order by a.time_order ');
CALL cr_debug.UPDATE_WATCH3('@v', @v, '', cr_stack_depth) ;
                              CALL cr_debug.TRACE(79, 79, 30, 50, cr_stack_depth) ;
PREPARE stm FROM @v;
                              CALL cr_debug.TRACE(80, 80, 30, 42, cr_stack_depth) ;
EXECUTE stm;
                              CALL cr_debug.TRACE(81, 81, 30, 53, cr_stack_depth) ;
DEALLOCATE PREPARE stm;                           CALL cr_debug.TRACE(81, 81, 80, 103, cr_stack_depth) ;
SET stErrorLog = '001';
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;


  

  
  CALL cr_debug.TRACE(87, 87, 2, 31, cr_stack_depth) ;
set iCumQTY_Start  = iCumQTY;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Start', iCumQTY_Start, '', cr_stack_depth) ;

  
  CALL cr_debug.TRACE(90, 90, 2, 15, cr_stack_depth) ;
OPEN curMain;
  CALL cr_debug.TRACE(91, 115, 2, 30, cr_stack_depth) ;
REPEAT
    CALL cr_debug.TRACE(92, 92, 4, 50, cr_stack_depth) ;
FETCH curMain INTO stVsPrimaryKey,iCumQTY_Raw;
CALL cr_debug.UPDATE_WATCH3('stVsPrimaryKey', stVsPrimaryKey, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Raw', iCumQTY_Raw, '', cr_stack_depth) ;
    CALL cr_debug.TRACE(93, 114, 4, 11, cr_stack_depth) ;
IF NOT doneCursor THEN                                  CALL cr_debug.TRACE(93, 93, 60, 103, cr_stack_depth) ;
SET stErrorLog = CONCAT('030_', stTool_Id);
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
      
      
         
      
      CALL cr_debug.TRACE(98, 109, 6, 13, cr_stack_depth) ;
IF iCumQTY_Start <> iCumQty_Raw  THEN                             CALL cr_debug.TRACE(98, 98, 72, 115, cr_stack_depth) ;
SET stErrorLog = CONCAT('031_', stTool_Id);
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;

        
        CALL cr_debug.TRACE(101, 101, 8, 80, cr_stack_depth) ;
INSERT INTO dcop_collect_tt1 ( VsPrimaryKey ) VALUES ( stVsPrimaryKey );
CALL cr_debug.UPDATE_SYSTEM_CALLS(102) ;

        
        CALL cr_debug.TRACE(104, 104, 8, 40, cr_stack_depth) ;
set iCumQTY_Start = iCumQTY_Raw;
CALL cr_debug.UPDATE_WATCH3('iCumQTY_Start', iCumQTY_Start, '', cr_stack_depth) ;

      
      ELSE                                                     CALL cr_debug.TRACE(107, 107, 63, 106, cr_stack_depth) ;
SET stErrorLog = CONCAT('033_', stTool_Id);
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
                                                              CALL cr_debug.TRACE(108, 108, 62, 105, cr_stack_depth) ;
SET stErrorLog = CONCAT('035_', stTool_Id);
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
      END IF;
                                                              CALL cr_debug.TRACE(110, 110, 62, 98, cr_stack_depth) ;
SET stErrorLog = CONCAT('050_', '');
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;

    
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CALL cr_debug.TRACE(116, 116, 2, 16, cr_stack_depth) ;
CLOSE curMain;

                                                       CALL cr_debug.TRACE(118, 118, 55, 78, cr_stack_depth) ;
SET stErrorLog = '900';
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ; 
                                                       CALL cr_debug.TRACE(119, 119, 55, 87, cr_stack_depth) ;
SELECT 1 INTO iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, '', cr_stack_depth) ; 
                                                       CALL cr_debug.TRACE(120, 120, 55, 80, cr_stack_depth) ;
SELECT iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ; 
                                                       CALL cr_debug.TRACE(121, 121, 55, 118, cr_stack_depth) ;
CALL sys_job_exec_sp(2, stJob_Name, dExec_Time, iElp_Spec, '');
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, '', cr_stack_depth) ;

CALL cr_debug.TRACE(123, 123, 0, 3, cr_stack_depth) ;
CALL cr_debug.LEAVE_MODULE(cr_stack_depth - 1) ;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `dcop_raw_capt_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `dcop_raw_capt_sp`(IN stDate_Re varchar(10))
BEGIN
   

  
  
  
  
  
  
  
  

  
  

  DECLARE stJob_Name varchar(32) DEFAULT "dcop_raw_capt_sp";
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  DECLARE stTool_Id  varchar(7) DEFAULT NULL; 
  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1    varchar(19) DEFAULT NULL;
  DECLARE stLast2    varchar(19) DEFAULT NULL;


  DECLARE doneCursor int DEFAULT 0;
  DECLARE curTool CURSOR FOR
     SELECT a.tool_id FROM oee_tool_bt a  WHERE 1 = 1  AND dcop_type = 2  ORDER BY a.tool_id; 
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;

                                                     DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                                       BEGIN
                                                         SELECT 0 INTO iResult_1_ok_0_ng;
                                                         ROLLBACK;
                                                         CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                         SELECT
                                                           iResult_1_ok_0_ng;
                                                       END;                                        
                                                       CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                                       SET stErrorLog = '000';
  
  
  


  
  IF LENGTH(stDate_Re) < 10 THEN 
    SET stDate_Re = NULL;
  END IF;

  
  delete from dcop_collect_tt1;
  

  
  OPEN curTool;
  REPEAT
    FETCH curTool INTO stTool_Id;
    IF NOT doneCursor THEN                                  SET stErrorLog = CONCAT('030_', stTool_Id);
      
      
      

      set stKeyField = "n16";


      
      IF stDate_Re IS NULL THEN                             SET stErrorLog = CONCAT('031_', stTool_Id);
        


        SELECT time_order, n16 INTO stLast1, iCumQTY
        FROM dcop_collect_ct
        WHERE 1 = 1
        AND tool_id = stTool_Id 
        AND time_order >= DATE_FORMAT(DATE_ADD(NOW(), INTERVAL -1 DAY), "%Y/%m/%d %H:%i:%S") 
        ORDER BY time_order
        LIMIT 1;

        
        
        
        
        

        
        SET stLast2 = DATE_FORMAT(NOW(), "%Y/%m/%d %H:%i:%S");

        
        IF stLast1 IS NULL THEN                                     SET stErrorLog = CONCAT('032_', stTool_Id);

          SET stLast1 = CONCAT(DATE_FORMAT(DATE_ADD(NOW(), INTERVAL -1 DAY), "%Y/%m/%d"), ' 00:00:00'); 
          SET stLast2 = CONCAT(DATE_FORMAT(NOW(), "%Y/%m/%d"), ' 59:59:59');
          SET iCumQTY = 0;

        END IF;

      
      ELSE                                                            SET stErrorLog = CONCAT('033_', stTool_Id);
        
        

        
        SET stLast1 = CONCAT(stDate_Re, ' 00:00:00');
        SET stLast2 = CONCAT(stDate_Re, ' 59:59:59');
        SET iCumQTY = 0;                                              SET stErrorLog = CONCAT('034_', stTool_Id);

        
        DELETE
          FROM dcop_collect_ct
        WHERE tool_id = stTool_Id
          AND time_order >= stLast1
          AND time_order <= stLast2;

                                                              SET stErrorLog = CONCAT('035_', stTool_Id);
      END IF;
                                                              SET stErrorLog = CONCAT('050_', '');



          SET stLast1 = '2019/11/23 09:00:00';
          SET stLast2 = '2019/11/23 09:59:59';
          SET iCumQTY = 0;


      
      CALL dcop_raw_capt_s1_sp(stDate_Re,stTool_Id, stLast1, stLast2,stKeyField,iCumQTY);  

    
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CLOSE curTool;                                       SET stErrorLog = CONCAT('061_', 'insert compact table');

 
  INSERT INTO dcop_collect_ct 
    select b.* from ( SELECT distinct VsPrimaryKey FROM dcop_collect_tt1 ) a JOIN dcop_collect_bt b
                 on ( a.VsPrimaryKey = b.VsPrimaryKey )
    order by b.tool_id asc, b.VsPrimaryKey ASC;        SET stErrorLog = CONCAT('061_', 'insert compact table-ok');

                                                       SET stErrorLog = '900'; 
                                                       SELECT 1 INTO iResult_1_ok_0_ng; 
                                                       SELECT iResult_1_ok_0_ng; 
                                                       CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `erp_stb_create_lot_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `erp_stb_create_lot_sp`( IN stMFG_NO VARCHAR(16) )
label_sp:
BEGIN
-- Tip:
-- 1. cancel mfg_no job, write in delphi, not need more stored procedure 2020/04/23 lkchena
-- 2.

-- execute
-- CALL erp_stb_create_lot_sp('MFG_NO-007');
-- SELECT * FROM ker_wip_bt a WHERE s= 'S' and mfg_no = 'MFG_NO-007'
-- Select * from ker_wip_tt;
-- SELECT * FROM sys_job_log_bt ORDER BY time_start desc

-- prepare sql 0:
-- DELETE FROM ker_wip_tt
-- DELETE FROM ker_wip_bt where s = 'S'

  declare  stJob_Name varchar(32) DEFAULT 'erp_stb_create_lot_sp';
  declare  iElp_Spec int DEFAULT 30;-- execute spec: seconds
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;-- return value, let delphi code can receive it 2017/11/27 lkchena
  declare  stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  
  DECLARE stPart_Id  varchar(64) DEFAULT null;
  DECLARE stRoute_Id varchar(64) DEFAULT null;
  DECLARE stMold_Id  varchar(64) DEFAULT null;
  DECLARE stTool_Id  varchar(64) DEFAULT null;
  DECLARE stOrder_No varchar(64) DEFAULT null;
  DECLARE iCnt_Plan   int        DEFAULT   0;
  DECLARE iBatch_Size int        DEFAULT   0;  
  DECLARE iBatch_Spec int        DEFAULT   0;-- vehichle total unit count spec
  DECLARE iPri        int        DEFAULT 350;
  
  DECLARE stDate_STB      varchar(10) DEFAULT null;
  DECLARE stDate_STB_Real varchar(10) DEFAULT null;
  DECLARE stDate_Due      varchar(10) DEFAULT null;
  DECLARE stDate_Late     varchar(10) DEFAULT null;
  DECLARE stDate_Comp     varchar(10) DEFAULT null;

  -- lot parameter
  declare stLOT_HEADER  varchar(1) DEFAULT null;-- default use one char to be header 
  declare stNODE_HEADER  varchar(2) DEFAULT null;-- two char for be flag 2020/04/22
  declare iLOT_LENGTH   int DEFAULT 6; 
  declare iLOT_COUNTER  int DEFAULT 0; 

  declare doneCursor INT DEFAULT 0;
  
/*
  declare cur1 CURSOR FOR 
     SELECT a.mfg_no, a.order_no, a.part_id, a.cust_id, a.date_stb, a.qty_order, a.qty_stock, a.qty_curr,
            b.lot_size_spec,b.cast_spec,b.cast_pcs_spec, b.powder_id
       FROM erp_order_mfg_bt a  LEFT JOIN dm_part_bt b ON a.part_id = b.part_id
     WHERE a.mfg_no LIKE sql_st_mfg_no AND
       ( ( a.f_create_lot = sql_if_create_lot1) OR ( a.f_create_lot = sql_if_create_lot2) ) -- for access different parameter -- good
      order BY a.mfg_no ASC;
 */ 
												  DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); -- 'error' iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        -- log start: 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');-- iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
                                                              set stErrorLog = '000';         
    -- 0.0 delete temp table
    set sql_safe_updates = 0; -- ??? why always show the error -- my god 2020/04/22 lkchena
    delete from ker_wip_tt; set stErrorLog = '001';         

    -- 0.1 get part
    select    part_id,   route_id,   mold_id,   tool_id,   order_no, cnt_plan,   pri,   date_stb,  date_due,  date_late,  date_comp -- no date_stb_real
      into  stPart_Id, stRoute_Id, stMold_Id, stTool_Id, stOrder_No, iCnt_Plan, iPri, stDate_STB,stDate_Due,stDate_Late,stDate_Comp
     from tool_order_bt
    where mfg_no = stMFG_No -- 'MFG_NO-007'
     limit 1; set stErrorLog = '010';         
                                                 IF stPart_Id is NULL THEN -- no setting 
                                                          SET stErrorLog = 'no part data'; signal sqlstate '45000' set message_text = stErrorLog;
                                                          leave label_sp;     
												 END IF;                                                  
    -- 1. get lot_size_spec
    select  lot_size_spec into iBatch_Spec
     from dm_part_bt
    where part_id = stPart_Id 
     limit 1;  set stErrorLog = '012';  -- select concat('lot_size_spec: ', iBatch_Spec);
     
    -- 1.2 get lot_num count  -- test code: set iCnt_Plan = 2500;
    select ( iCnt_Plan Div iBatch_Spec ) into iTmp;
    if iTmp = 0 then
       set iTmp = 1;
    else       
      if mod(iCnt_Plan,iBatch_Spec) <> 0 then
        set iTmp = iTmp + 1;
      end if;        
    end if; set stErrorLog = '015';  -- select concat(iCnt_Plan,'/',iBatch_Spec,' iTmp: ', iTmp);
    
   -- 1.3 get lot parameter
   SELECT value1,value2,ivalue1,ivalue2  into stLOT_HEADER,stNODE_HEADER,iLOT_LENGTH,iLOT_COUNTER
    FROM sys_param_conf_bt a WHERE param_id = 'LOT-01'; set stErrorLog = '018'; -- select concat(stLOT_HEADER,' / ',stNODE_HEADER,' / ',iLOT_LENGTH,' / ',iLOT_COUNTER);
    
   -- 1.5 cum lot  count
   update sys_param_conf_bt 
     set ivalue2 = ivalue2 + iTmp
   WHERE param_id = 'LOT-01'; set stErrorLog = '020';
               
                             
    -- 2. run iPlan cycle
WHILE ( iCnt_Plan > 0 ) or ( ( iCnt_Plan > 0)and((iBatch_Spec-iCnt_Plan)>0) )  DO   
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '200';
	-- 2.1
    set iBatch_Size = iBatch_Spec;
    if (iBatch_Spec-iCnt_Plan)>0 then
      set iBatch_Size = iCnt_Plan;
    end if; -- debug: select iBatch_Size; -- iCnt_Plan;
    
    -- 2.6 insert wip -- get lot_id
    -- select CONCAT(stLOT_HEADER,stNODE_HEADER,LPAD(iLOT_COUNTER,iLOT_LENGTH, '0'),'.0' ) AS lot_id;
      
    insert into ker_wip_tt
            (report_time,cate,lot_id,lot_id_p1,lot_id_p2,lot_id_p3,cart_no,box_no,lot_size,box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,
             s,pri,part_id,part_raw,route_id,raw_2d_code,ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,
             claim_time,claim_user,tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
             area_id,area_name,pos_id,pos_desc,tool_id,tag_track_io,proc_time,cust_id,cust_name,order_no,mfg_no,rd_flag,
             b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
             mold_id,powder_id,powder_type,cnt_plan,cnt_cur,cnt_act,cnt_ng,cnt_qc,cnt_test,cnt_tool,cnt_empty,
             date_stb,date_stb_real,date_due,date_late,date_comp,
             split_from,merge_from,full_num,full_code,full_time,full_2d)
            SELECT 
                -- 1
                now() as report_time,
                'RT' as cate,
                CONCAT(stLOT_HEADER,stNODE_HEADER,LPAD(iLOT_COUNTER,iLOT_LENGTH, '0'),'.0' ) as lot_id,
                CONCAT(stLOT_HEADER,stNODE_HEADER,LPAD(iLOT_COUNTER,iLOT_LENGTH, '0'),'.0' ) as lot_id_p1,
                null as lot_id_p2,
                null as lot_id_p3,
                null as cart_no,
                null as box_no,
                iBatch_Size as lot_size,
                0 as box_cnt,
                iBatch_Spec as lot_size_spec,
                0 as box_size_spec,
                0 as wafer_size_spec,
                'S' as s,
                -- 2
                iPri as pri,
                part_id,
                part_raw,
                route_id,
                null as raw_2d_code,
                ope_no,
                ope_name,
                stage_id,
                stage_name,
                stage_order,
                'stb' as ope_cate,
                0 as extra_step,
                -- 3
                now() as claim_time,
                'SYS' as claim_user,
                tool_grp_id,
                tool_grp,
                ws_type,
                null as ws_func,
                tool_type1,
                tool_type2,
                tool_type3,
                in_out,
                null as tool_func,
                null as er,
                -- 4
                area_id,
                area_name,
                null as pos_id,
                null as pos_desc,
                stTool_Id as tool_id, -- 'AF35Y01'
                null as tag_track_io,
                proc_time,
                cust_id,
                cust_name,
                stOrder_No as order_no,
                stMFG_No as mfg_no,
                rd_flag,
                -- 5
                null as b_vendor_id,
                null as b_vendor_name,
                'act: stb lot'lot_note,
                null as lot_memo,
                null as track_in_time,
                null as track_out_time,
                null as proc_start_time,
                null as proc_end_time,
                -- 6
                mold_id,
                null as powder_id,
                powder_type,
                iBatch_Size as cnt_plan,
                0 as cnt_cur,
                0 as cnt_act,
                0 as cnt_ng,
                0 as cnt_qc,
                0 as cnt_test,
                0 as cnt_tool,
                0 as cnt_empty,
                -- 7
                stDate_STB,stDate_STB_Real,stDate_Due,stDate_Late,stDate_Comp, -- 2020/04/23 lkchena  
                -- 8
                null as split_from,
                null as merge_from,
                null as full_num,
                null as full_code,
                null as full_time,
                null as full_2d
                -- ,node_id,node_time
            FROM -- mes_omi.ker_wip_bt;
            dm_flow_bv
            where 1=1
             and part_id = stPart_Id -- 'ACQ51F-1'
             and ope_no > '065.020'
             limit 1;
   
    -- 2.9 
    set iCnt_Plan = iCnt_Plan - iBatch_Spec;
    set iLOT_COUNTER = iLOT_COUNTER + 1;          SET stErrorLog = '299';    
   -- ---------------------------------------------------------------------------
END WHILE;                                        SET stErrorLog = '300';    

    -- 3.1 join stb wip into ker_wip_bt
    delete from  ker_wip_bt where mfg_no = stMFG_No and s = 'S'; -- only delete wait run stb lot 
                                                  SET stErrorLog = '310';    
    -- 3.2 insert to ker_wip_bt 
    insert into ker_wip_bt select * from ker_wip_tt;
    

    -- 9.0 final
                                                         set stErrorLog = '900';  
                                                         -- log execution time
                                                         SELECT 1 into iResult_1_ok_0_ng; -- return procedure ok or not
                                                         SELECT iResult_1_ok_0_ng; -- return value, let delphi code can receive it 2017/11/27 lkchena                                                                      
                                                               call sys_job_exec_sp( 9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog);-- log end:  -- iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `ker_sum_10mins_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `ker_sum_10mins_sp`(
   stCate varchar(6) )
BEGIN


declare  iResult_1_ok_0_ng INT;


 
 delete a.* from ker_sum_st a where a.cate = stCate;
 
 
 insert into ker_sum_st
 select 
  now() as `report_time`,
  stCate as `cate`, 
  a.area_id as `area_id`,
  b.area_name as `area_name`,
  b.area_order as `area_order`,
  a.eqp_grp_id as `eqp_grp_id`,
  a.eqp_grp as `eqp_grp`,
  a.eqp_grp_order as `eqp_grp_order`,
  0 as `demand`,
  0 as `capacity`,
  0 as `avl`,
  0 as `eff`,
  0 as `lost`,
  0 as `pwip`,
  COALESCE(c.rwip + c.qwip + c.hwip,0) as `wip`, 
  0 as `move`,
  COALESCE(c.qwip,0)  as `qwip`, 
  0 as `qtime`,
  COALESCE(c.rwip,0) as `rwip`,
  0 as `rtime`,
  COALESCE(c.hwip,0) as `hwip`,
  0 as `htime`,
  0 as `backup`,
  0 as `move_d`,
  0 as `move_n`,
  0 as `move_o`



from ker_eqp_grp_bt a  left join 
       ( 
         select a.eqp_grp_id, sum(rwip) as rwip, sum(qwip) as qwip, sum(hwip) as hwip 
          from
          ( select a.eqp_grp_id, 
                 if(a.s='R',a.lot_size ,0) as rwip, 
				 if(a.s='Q',a.lot_size ,0) as qwip, 
                 if(a.s='H',a.lot_size ,0) as Hwip
               from ker_wip_rt a
           ) a
           group by a.eqp_grp_id     
     
     )  as c on a.eqp_grp_id = c.eqp_grp_id, area_conf_bt b     
where 1=1
 and b.area_id = a.area_id;
 

 SELECT 1 into iResult_1_ok_0_ng; 
 SELECT iResult_1_ok_0_ng; 



END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `ker_wip_snapshot_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `ker_wip_snapshot_sp`(
   stCate varchar(6), dReportTime datetime )
BEGIN















  declare  stJob_Name varchar(32) DEFAULT 'ker_wip_snapshot_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();
  
  declare  iResult_1_ok_0_ng INT;

                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
                                                      SELECT 0 into iResult_1_ok_0_ng; 
                                                      ROLLBACK;     
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,'error');
                                                      SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');



 
 START TRANSACTION;
 delete a.* from ker_wip_tt a where a.cate = stCate;
 COMMIT;
 
 
 START TRANSACTION;
  INSERT into ker_wip_tt 
   SELECT dReportTime AS report_time,stCate,
          lot_id ,cart_id ,lot_size ,piece_size ,cast_pcs_spec,
          s ,pri ,part_id ,part_raw ,raw_2d_code ,ope_no ,ope_name,
          stage_id ,stage_name ,stage_order ,ope_cate ,claim_time ,eqp_grp_id ,eqp_grp ,er,
          area_id ,area_name ,tool_id ,cust_id ,cust_name ,order_id ,b_vendor_id ,b_vendor_name,
          lot_note ,lot_memo ,track_in_time ,track_out_time ,proc_start_time ,proc_end_time,mold_id 
   FROM ker_wip_rt a;
 COMMIT;

 
 START TRANSACTION;
  DELETE FROM ker_wip_bth where report_time = dReportTime and cate = stCate;
  INSERT INTO ker_wip_bth SELECT a.* FROM ker_wip_tt a where a.cate = stCate;
 COMMIT;
 
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `ksr_sum_10mins_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `ksr_sum_10mins_sp`(
   stCate varchar(6) )
BEGIN





 
 delete a.* from ksr_sum_st a where a.cate = stCate;
 
 insert into ksr_sum_st
 select 
    now() as `report_time`,  
    stCate as `cate`, 
    2 as d_type,
    a.area_id as `area_id`,
    b.area_name as `area_name`,
    b.area_order as `area_order`,
    a.stage_id as `stage_id`,
    a.stage_name as `stage_name`,
    a.stage_order as `stage_order`,
    0 as `demand`,
    0 as `pwip`,
    COALESCE(c.rwip + c.qwip + c.hwip,0) as `wip`, 
    0 as `move`,
    COALESCE(c.qwip,0)  as `qwip`, 
    0 as `qtime`,
    COALESCE(c.rwip,0) as `rwip`,
    0 as `rtime`,
    COALESCE(c.hwip,0) as `hwip`,
    0 as `htime`,
    0 as `backup`,
    0 as `move_d`,
    0 as `move_n`,
    0 as `move_o`
  from dm_stage_bt a  left join 
       ( 
         select a.stage_id, sum(rwip) as rwip, sum(qwip) as qwip, sum(hwip) as hwip 
          from
          ( select a.stage_id, 
                 if(a.s='R',a.lot_size ,0) as rwip, 
				 if(a.s='Q',a.lot_size ,0) as qwip, 
                 if(a.s='H',a.lot_size ,0) as Hwip
               from ker_wip_rt a
           ) a
           group by a.stage_id    
     
     )  as c on a.stage_id = c.stage_id, area_conf_bt b     
  where 1=1
   and b.area_id = a.area_id;
 



END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `log_hist_del_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `log_hist_del_sp`(
	IN `iLog_Hist_day` INT



)
    COMMENT 'log history delete  exceed day data'
BEGIN
  
 declare  stJob_Name varchar(32) DEFAULT 'log_hist_del_sp';
 declare  iElp_Spec int DEFAULT 30;
 declare  dExec_Time datetime DEFAULT NOW();  
 declare  stErrorLog varchar(64);
 declare  iResult_1_ok_0_ng INT;   


     
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
																				  
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              
                                                              select '000' INTO stErrorLog; 
                                                                        													                                             
                                                              
 
 DELETE FROM sys_job_log_bth 
 where time_start <= DATE_SUB(NOW(),INTERVAL iLog_Hist_day DAY);                                                            
 
                                                              
                                                              select '900' INTO stErrorLog;
  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');
																					


  
  

  
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `log_move_hist_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8mb4 */ ;
/*!50003 SET character_set_results = utf8mb4 */ ;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `log_move_hist_sp`(
	IN `iLog_keep_day` INT




)
    COMMENT 'only keep ??? days, then move to log history table'
BEGIN
  
 declare  stJob_Name varchar(32) DEFAULT 'log_move_hist_sp';
 declare  iElp_Spec int DEFAULT 30;
 declare  dExec_Time datetime DEFAULT NOW();  
 declare  stErrorLog varchar(64);
 declare  iResult_1_ok_0_ng INT;   


     
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
																				  
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              
                                                                         													                                                                                                                 select '000' INTO stErrorLog; 
  
  INSERT sys_job_log_bth  
  SELECT * FROM sys_job_log_bt a 
  where 1=1 and time_start <= DATE_SUB(NOW(),INTERVAL iLog_keep_day DAY);                                                                      													                                                                                                                															  
                                                              select '001' INTO stErrorLog;
 
  DELETE FROM sys_job_log_bt 
  where time_start <= DATE_SUB(NOW(),INTERVAL iLog_keep_day DAY);
  
                               
                                                              
                                                           
 
                                                              
                                                              select '900' INTO stErrorLog;
  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');
																					


  
  

  
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `mfg_laser_cr8_lot_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `mfg_laser_cr8_lot_sp`( 
 stMFG_NO VARCHAR(64), stOpe_No varchar(64), stS varchar(64), stCart_No varchar(64), stTool_Id varchar(64),
 iLot_Size int, iBox_Size int, iWafer_Size int, iWafer_Level int )
label_sp:
BEGIN
-- 2020/04/29 lkchena: for laser marking tool

-- parameter:
-- iWafer_Level: 0:normal, no wafer_level data(ker_wip_w0_bt) 2:need wafer level, write wafer data (ker_wip_w2_bt)

-- execute
/*
 CALL mfg_laser_cr8_lot_sp('MFG_NO-107','860.100','Q','ACS002','AZMKD01',2100,20,105,2);
 CALL mfg_laser_cr8_lot_sp('MFG_NO-107','860.100','Q','ACS002','AZMKD01',2100,20,105,0);
 CALL mfg_laser_cr8_lot_sp('MFG_NO-107','','','','',0,0,0);
 SELECT * FROM ker_wip_bt a WHERE mfg_no = 'MFG_NO-107' and ope_no = '860.100'
 Select * from ker_wip_tt;
 SELECT * FROM sys_job_log_bt ORDER BY time_start desc
*/
-- prepare sql 0:
-- DELETE FROM ker_wip_tt
-- DELETE FROM ker_wip_bt where s = 'S'

  declare  stJob_Name varchar(32) DEFAULT 'mfg_laser_cr8_lot_sp';
  declare  iElp_Spec int DEFAULT 30;-- execute spec: seconds
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;-- return value, let delphi code can receive it 2017/11/27 lkchena
  declare  stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  
  DECLARE stPart_Id   varchar(64) DEFAULT null;
  DECLARE stRoute_Id  varchar(64) DEFAULT null;
  DECLARE stMold_Id   varchar(64) DEFAULT null;
  DECLARE stOrder_No  varchar(64) DEFAULT null;
  
  DECLARE stOpe_No2 varchar(7)   DEFAULT null;
  DECLARE stOpe_Start varchar(7) DEFAULT null;
  DECLARE stOpe_End   varchar(7) DEFAULT null;
  DECLARE stS2        varchar(1) DEFAULT null;
  DECLARE stCart_No2  varchar(64) DEFAULT null;  
  
  DECLARE iLot_Size2    int       DEFAULT   0;
  DECLARE iBox_Size2    int       DEFAULT   0;
  DECLARE iWafer_Size2  int       DEFAULT   0;
  
  DECLARE iCnt_Plan   int        DEFAULT   0;
  DECLARE iBatch_Size int        DEFAULT   0;  
  DECLARE iBatch_Spec int        DEFAULT   0;-- vehichle total unit count spec
  DECLARE iPri        int        DEFAULT 350;
  
  DECLARE stDate_STB      varchar(10) DEFAULT null;
  DECLARE stDate_STB_Real varchar(10) DEFAULT null;
  DECLARE stDate_Due      varchar(10) DEFAULT null;
  DECLARE stDate_Late     varchar(10) DEFAULT null;
  DECLARE stDate_Comp     varchar(10) DEFAULT null;
  DECLARE iUnit_Type      int         DEFAULT 0; -- 2020/05/02 LKCHENA 

  -- lot parameter
  declare stLot_Id       varchar(64) DEFAULT null;
  declare stLOT_HEADER   varchar(1) DEFAULT null;-- default use one char to be header 
  declare stNODE_HEADER  varchar(2) DEFAULT null;-- two char for be flag 2020/04/22
  declare iLOT_LENGTH    int DEFAULT 6; 
  declare iLOT_COUNTER   int DEFAULT 0; 

  declare doneCursor INT DEFAULT 0;  
												  DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); -- 'error' iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        -- log start: 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');-- iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
                                                              set stErrorLog = '000';         															
    -- 0.0 delete temp table
    -- 2020/04/29 lkchena 
    -- dbfoge can't debug, when break at 
    set sql_safe_updates = 0; -- ??? why always show the error -- my god 2020/04/22 lkchena
    delete from ker_wip_tt;         
    set stErrorLog = '001';         

    -- 0.1 get parameter
     SELECT value1,value2 into stOpe_Start, stOpe_End 
      FROM sys_param_conf_bt 
     where param_id = 'FLOW-01';    set stErrorLog = '005'; -- select stOpe_Start, stOpe_End;
     
    -- 0.3 get cart_no
     -- stCart_No null
     set stCart_No2 = stCart_No;
     
    -- 0.4 deal ope_no
     set stOpe_No2 = stOpe_No;
     if LENGTH(stOpe_No) = 0 then
       set stOpe_No2 = stOpe_Start;     
     end if;                        set stErrorLog = '011'; -- select stOpe_No,stOpe_No2;
     
    -- 0.5 get s
     if LENGTH(stS) = 0 then
          set stS2 = 'S';
     else set stS2 = stS;
     end if;                        set stErrorLog = '015';  -- select stS,stS2;
     
    -- 0.6 get part -- here no tool_id -- 2020/04/29 lkchena
  	select    part_id,   route_id,   mold_id,   order_no, cnt_plan,   pri,   date_stb,  date_due,  date_late,  date_comp, unit_type,  lot_size_spec,box_size_spec,wafer_size_spec -- no date_stb_real
      into  stPart_Id, stRoute_Id, stMold_Id, stOrder_No, iCnt_Plan, iPri, stDate_STB,stDate_Due,stDate_Late,stDate_Comp, iUnit_Type, iLot_Size2, iBox_Size2, iWafer_Size2
     from mfg_order_bt 
    where mfg_no = stMFG_No;         set stErrorLog = '016'; -- 'MFG_NO-107'
         
	  -- 0.7 get part related parameter 
    if iLot_Size <> 0 then -- 沒填 就使用預設值
    
      SET iLot_Size2   = iLot_Size;
      SET iBox_Size2   = iBox_Size; 
      SET iWafer_Size2 = iWafer_Size;
    
    end if;                          set stErrorLog = '017'; -- select iLot_Size, iBox_Cnt, iBox_Size, iLot_Size2, iBox_Cnt2, iBox_Size2;
  
  	-- 1.0.1 get lot_id parameter
    SELECT value1,value2,ivalue1,ivalue2  into stLOT_HEADER,stNODE_HEADER,iLOT_LENGTH,iLOT_COUNTER
    FROM sys_param_conf_bt a WHERE param_id = 'LOT-01'; set stErrorLog = '031'; -- select concat(stLOT_HEADER,' / ',stNODE_HEADER,' / ',iLOT_LENGTH,' / ',iLOT_COUNTER);

  	-- 1.0.2 update lot cumlate number
    update sys_param_conf_bt set ivalue2 = ivalue2 + 1
    WHERE param_id = 'LOT-01'; set stErrorLog = '035'; -- select concat(stLOT_HEADER,' / ',stNODE_HEADER,' / ',iLOT_LENGTH,' / ',iLOT_COUNTER);

    -- 1.0.9 get lot_id
    select fn_cr8_lot_id(stLOT_HEADER,stNODE_HEADER,iLOT_COUNTER,iLOT_LENGTH) into stLot_Id; set stErrorLog = '038';    

    -- 1.1 insert wip
    insert into ker_wip_tt
            (report_time,cate,lot_id,lot_id_p1,lot_id_p2,lot_id_p3,cart_no,box_no,lot_size,box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,box_label,
             s,pri,part_id,part_raw,route_id,raw_2d_code,ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,
             claim_time,claim_user,tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
             area_id,area_name,pos_id,pos_desc,tool_id,tag_track_io,proc_time,cust_id,cust_name,order_no,mfg_no,rd_flag,
             b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
             mold_id,powder_id,powder_type,cnt_plan,cnt_cur,cnt_act,cnt_ng,cnt_qc,cnt_test,cnt_tool,cnt_empty,
             date_stb,date_stb_real,date_due,date_late,date_comp,
             split_from,merge_from,full_id,full_time,full_2d)
            SELECT 
                -- 1
                now() as report_time,
                'RT' as cate,
                stLot_Id AS lot_id,
                stLot_Id AS lot_id_p1,
                null as lot_id_p2,
                null as lot_id_p3,
                stCart_No2 as cart_no,
                null as box_no,
                iLot_Size2 as lot_size,
                iBox_Size2 as box_cnt,
                iLot_Size2 as lot_size_spec,
                iBox_Size2 as box_size_spec,
                iWafer_Size2 as wafer_size_spec,
                null as box_label,
                stS2 as s,
                -- 2
                iPri as pri,
                part_id,
                part_raw,
                route_id,
                null as raw_2d_code,
                ope_no,
                ope_name,
                stage_id,
                stage_name,
                stage_order,
                'stb' as ope_cate,
                0 as extra_step,
                -- 3
                now() as claim_time,
                'SYS' as claim_user,
                tool_grp_id,
                tool_grp,
                ws_type,
                null as ws_func,
                tool_type1,
                tool_type2,
                tool_type3,
                in_out,
                null as tool_func,
                null as er,
                -- 4
                area_id,
                area_name,
                null as pos_id,
                null as pos_desc,
                stTool_Id as tool_id, -- 'AF35Y01'
                null as tag_track_io,
                proc_time,
                cust_id,
                cust_name,
                stOpe_No2 as order_no,
                stMFG_No as mfg_no,
                rd_flag,
                -- 5
                null as b_vendor_id,
                null as b_vendor_name,
                'act: laser marking lot'lot_note,
                null as lot_memo,
                now() as track_in_time,
                date_add(now(), interval 1 second) as track_out_time,
                null as proc_start_time,
                null as proc_end_time,
                -- 6
                mold_id,
                null as powder_id,
                powder_type,
                iLot_Size2 as cnt_plan,
                iLot_Size2 as cnt_cur,
                iLot_Size2 as cnt_act,
                0 as cnt_ng,
                0 as cnt_qc,
                0 as cnt_test,
                iLot_Size2 as cnt_tool,
                0 as cnt_empty,
                -- 7
                stDate_STB,stDate_STB_Real,stDate_Due,stDate_Late,stDate_Comp, -- 2020/04/23 lkchena  
                -- 8
                null as split_from,
                null as merge_from,
                null as full_id,
                null as full_time,
                null as full_2d
                -- ,node_id,node_time
            FROM -- mes_omi.ker_wip_bt;
            dm_flow_bv
            where 1=1
             and part_id = stPart_Id -- 'ACQ51F-1'
             and ope_no = stOpe_No2 -- '860.100'
             limit 1;                 set stErrorLog = '190';
    
    -- 3.0 keep ker_wip_tt data, let sub-procedure execute it
    call mfg_laser_cr8_s_w_sp(stMFG_NO,stLot_Id,iLot_Size,iBox_Size,iWafer_Size,iUnit_Type,iWafer_Level); set stErrorLog = '290';
    
    -- 6.1 write temp data to real table
    delete from  ker_wip_bt where lot_id = stLot_Id; -- only delete wait run stb lot 
    insert into ker_wip_bt select * from ker_wip_tt;
                                                         SET stErrorLog = '610';    
    
                                                         set stErrorLog = '900';  
                                                         -- log execution time
                                                         SELECT 1 into iResult_1_ok_0_ng; -- return procedure ok or not
                                                         SELECT iResult_1_ok_0_ng; -- return value, let delphi code can receive it 2017/11/27 lkchena                                                                      
                                                               call sys_job_exec_sp( 9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog);-- log end:  -- iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `mfg_laser_cr8_s_w_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `mfg_laser_cr8_s_w_sp`( 
 stMFG_NO VARCHAR(64), stLot_Id VARCHAR(64), iLot_Size int, iBox_Size int, iWafer_Size int, iUnit_Type int, iWafer_Level int )
label_sp:
BEGIN
-- 2020/04/29 lkchena: for laser marking tool

-- parameter:
-- iWafer_Level: 0:normal, no wafer_level data(ker_wip_w0_bt) 2:need wafer level, write wafer data (ker_wip_w2_bt)
-- execute
/*
 CALL mfg_laser_cr8_s_w_sp('MFG_NO-107','A01000337.0',2100,20,105,0,2);
 CALL mfg_laser_cr8_s_w_sp('MFG_NO-107','A01000337.0',2100,20,105,0,0);
 CALL mfg_laser_cr8_s_w_sp('MFG_NO-107','A01000337.0',2100,20,105,1);

 SELECT  a.full_id,a.full_time,a.full_2d, a.* FROM ker_wip_w2_bt a  order by report_time desc;
 SELECT * FROM ker_wip_w2_bt a  order by report_time desc;
 
 SELECT  a.full_id,a.full_time,a.full_2d, a.* FROM ker_wip_w0_bt a  order by report_time desc;
 SELECT * FROM ker_wip_w0_bt a  order by report_time desc;

 Select * from ker_wip_tt;
 SELECT * FROM sys_job_log_bt ORDER BY time_start desc;
*/
  declare  stJob_Name varchar(32) DEFAULT 'mfg_laser_cr8_s_w_sp';
  declare  iElp_Spec int DEFAULT 30;-- execute spec: seconds
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;-- return value, let delphi code can receive it 2017/11/27 lkchena
  declare  stErrorLog varchar(64);

  DECLARE iTmp  int DEFAULT 0;
  DECLARE iCNT  int DEFAULT 0;
  DECLARE yTmp  int DEFAULT 0;
  DECLARE yTmp2 int DEFAULT 0; -- for doulbe loop 2020/05/02 lkchena
  
  declare stFull_Id   varchar(3072) DEFAULT NULL;
  declare stFull_Time varchar(3072) DEFAULT NULL;
  declare stFull_2d   varchar(3072) DEFAULT NULL;

  declare stBox_Id    VARCHAR(64) DEFAULT NULL;  
  declare stBox_No    VARCHAR(64) DEFAULT NULL;  
  declare stWafer_Id  VARCHAR(64) DEFAULT NULL;  
  declare stBox_Label VARCHAR(64) DEFAULT NULL;  -- 標籤紙號碼 2020/05/03 lkchen

  DECLARE iLoop int DEFAULT 0;
  
  declare stLOT_HEADER   varchar(1) DEFAULT null;-- default use one char to be header 
  declare stNODE_HEADER  varchar(2) DEFAULT null;-- two char for be flag 2020/04/22
  declare iLOT_LEN       int DEFAULT 5; 
  declare iLOT_CUM       int DEFAULT 0; 
  declare stLOT_LEAD     varchar(64) DEFAULT null;-- remove .XXX -- 2020/05/02 LKCHENA

  -- lot parameter
  declare doneCursor INT DEFAULT 0;  
												  DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); -- 'error' iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        -- log start: 
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');-- iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
                                                              set stErrorLog = '000';         		
                                                              
    -- 0.0 delete temp table
    -- 2020/04/29 lkchena 
    -- dbfoge can't debug, when break at
    select @@global.sql_safe_updates into iTmp;
    if iTmp <> 0 then set sql_safe_updates = 0; end if; -- ??? why always show the error -- my god 2020/04/22 lkchena
    
	 IF     iWafer_Level = 2 THEN   -- need wafer data 
            
              delete from ker_wip_w2_tt;   
                                                  
	 ELSEIF iWafer_Level = 0 THEN  -- normal omi

               delete from ker_wip_w0_tt;   
     -- ELSE 
     END IF;                                set stErrorLog = '001';     
    
    -- 0.1 get parameter
     set stLOT_LEAD = substr(stLot_Id,1,length(stLot_Id)-2); set stErrorLog = '010'; -- select stLOT_LEAD; -- remove .XXX <-- 'A01000337'  'A01000337.0' -- 2020/05/02 LKCHENA
                         				
    -- 1.1a update / get wafer_id cumlated number
    SELECT value1,value2,ivalue1,ivalue2  into stLOT_HEADER,stNODE_HEADER,iLOT_LEN,iLOT_CUM
    FROM sys_param_conf_bt a WHERE param_id = 'BOX-90'; set stErrorLog = '031'; -- select concat(stLOT_HEADER,' / ',stNODE_HEADER,' / ',iLOT_LEN,' / ',iLOT_CUM);

  	-- 1.1b update lot cumlate number
    update sys_param_conf_bt set ivalue2 = ivalue2 + iBox_Size -- different
    WHERE param_id = 'BOX-90'; set stErrorLog = '035'; -- select concat(stLOT_HEADER,' / ',stNODE_HEADER,' / ',iLOT_LEN,' / ',iLOT_CUM);
          
    -- 3.0 
    set iTmp = 1;-- from 1, .00 is rerved 2020/05/02 lkchena
    set yTmp = 1; -- from 1, .000 is rerved 2020/05/02 lkchena
    label_loop: LOOP           set stErrorLog = '310';     
      
       -- 2.1 get ld
       set stBox_Id  = fn_cr8_box_id(stLot_LEAD,iTmp,0);    set stErrorLog = '312';
-- qqq, wait code -- 2020/05/03 lkchena
       set stBox_Label  = '';  set stErrorLog = '313';
	
       -- 2.2 get box_no : here is virtual box_id, still need indepandant number
	   select fn_cr8_dum_box_no(stLOT_HEADER,iLOT_CUM,iLOT_LEN) into stBox_No; set stErrorLog = '038';    
       set iLOT_CUM = iLOT_CUM + 1; -- for create next wafer - 2020/05/02 lkchena

       -- 2.5 run wafer cycle to get full_xxx
       -- -----------------------------------------------------------------------------
       -- -----------------------------------------------------------------------------
              set stFull_Id=''; set stFull_Time=''; set stFull_2d='';
              
              -- move to outside 
              -- set yTmp = 1; -- from 1, .000 is rerved 2020/05/02 lkchena
              set yTmp2 = 1; -- for double loop -- 2020/05/02 lkchena
              label_yloop: LOOP                                 set stErrorLog = '510';

                 -- 2.3 full_code,full_time,full_2d
                 if iUnit_Type = 0 then -- by piece
										                        set stErrorLog = '520';
                      -- 2.2b stFull_Id: key: here is yTmp, not yTmp2
                      select fn_cr8_wafer_id(stLOT_LEAD,yTmp,iUnit_Type) into stWafer_Id;
                      set stFull_Id = concat(stFull_Id,',',stWafer_Id ); -- wafer count: max 999 in one box
                                                                set stErrorLog = '530';
                    -- 2.2c stFull_Time
					  set stFull_Time = concat(stFull_Time,',', DATE_FORMAT( now(), "%Y%m%d%H%i%S") ); -- save db laoding 2020/05/04
					  -- old: set stFull_Time = concat(stFull_Time,',', DATE_FORMAT( now(), "%Y/%m/%d %H:%i:%S") );
                                                                set stErrorLog = '540';
-- qqq: wait code -- 2020/05/03 lkchena
                      -- 2.2d stFull_2d
                      set stFull_2d = '';                       set stErrorLog = '550'; 

                  else -- by weight or others

                      set stFull_Id   = '';
                      set stFull_Time = '';
                      set stFull_2d   = '';

                  end if;
													        set stErrorLog = '590';
                -- key: yTmp2, not yTmp                                            
                SET yTmp  = yTmp  + 1;
                SET yTmp2 = yTmp2 + 1;
                IF( yTmp2 > iWafer_Size ) THEN LEAVE label_yloop; END IF;
            END LOOP;                                       set stErrorLog = '599';
       -- -----------------------------------------------------------------------------
       -- -----------------------------------------------------------------------------
       
       -- 2.9 insert box_id
		IF iWafer_Level = 2 THEN          set stErrorLog = '601'; -- need wafer data 

                   insert into ker_wip_w2_tt
                       (     report_time,cate,lot_id,lot_id_p1,lot_id_p2,lot_id_p3,
                             cart_no,box_no,lot_size,box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,box_label,s,pri,part_id,part_raw,route_id,raw_2d_code,
                             ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,claim_time,claim_user,
                             tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                             area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                             mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                             mold_id,powder_id,powder_type,cnt_plan,cnt_cur,cnt_act,cnt_ng,cnt_qc,cnt_test,cnt_tool,cnt_empty,
                             date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                             full_id,full_time,full_2d
						)
                  SELECT report_time,cate,
                           stBox_Id as lot_id, stLot_Id as lot_id_p1, stBox_Id as lot_id_p2, lot_id_p3,-- 'A01000337.00'
                                 cart_no,
                           stBox_No as box_no,
                           iWafer_Size as lot_size,
                                 box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,
                           stBox_Label as box_label,
                                 s,pri,part_id,part_raw,route_id,raw_2d_code,
                                 ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,claim_time,claim_user,
                                 tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                                 area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                                 mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                                 mold_id,powder_id,powder_type,
                           -- //keep wip information -- 2020/05/02 lkchena-- cnt_plan,cnt_cur,cnt_act, cnt_ng,cnt_qc,cnt_test,
						   -- use current wafer_size 2020/05/05 lkchena
                           iWafer_Size as cnt_plan, iWafer_Size as cnt_cur, iWafer_Size as cnt_act,
                                  cnt_ng,cnt_qc,cnt_test,cnt_tool, cnt_empty,
                                 date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                           stFull_Id,stFull_Time,stFull_2d
                   FROM ker_wip_tt limit 1;                                 set stErrorLog = '609';
     
		ELSEIF iWafer_Level = 0 THEN      set stErrorLog = '620';-- normal omi

            set stFull_Id   = null;
            set stFull_Time = null;
            set stFull_2d   = null;

                   insert into ker_wip_w0_tt
                       (     report_time,cate,lot_id,lot_id_p1,lot_id_p2,lot_id_p3,
                             cart_no,box_no,lot_size,box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,box_label,s,pri,part_id,part_raw,route_id,raw_2d_code,
                             ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,claim_time,claim_user,
                             tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                             area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                             mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                             mold_id,powder_id,powder_type,cnt_plan,cnt_cur,cnt_act,cnt_ng,cnt_qc,cnt_test,cnt_tool,cnt_empty,
                             date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                             full_id,full_time,full_2d
						)
                  SELECT report_time,cate,
                           stBox_Id as lot_id, stLot_Id as lot_id_p1, stBox_Id as lot_id_p2, lot_id_p3,-- 'A01000337.00'
                                 cart_no,
                           stBox_No as box_no,
                           iWafer_Size as lot_size,
                                 box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,
                           stBox_Label as box_label,
                                 s,pri,part_id,part_raw,route_id,raw_2d_code,
                                 ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,claim_time,claim_user,
                                 tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                                 area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                                 mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                                 mold_id,powder_id,powder_type,
                           -- //keep wip information -- 2020/05/02 lkchena-- cnt_plan,cnt_cur,cnt_act, cnt_ng,cnt_qc,cnt_test,
						   -- use current wafer_size 2020/05/05 lkchena
                           iWafer_Size as cnt_plan, iWafer_Size as cnt_cur, iWafer_Size as cnt_act,
                                 cnt_ng,cnt_qc,cnt_test,cnt_tool, cnt_empty,
                                 date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                          stFull_Id,stFull_Time,stFull_2d
                   FROM ker_wip_tt limit 1;                                 set stErrorLog = '619';

        -- ELSE 
        END IF;                           set stErrorLog = '699';
			
	  SET iTmp = iTmp + 1;
      IF( iTmp > iBox_Size ) THEN LEAVE label_loop; END IF;           -- select concat('iLoop: ',iLoop);
	END LOOP;                                                       set stErrorLog = '700';  
        
    -- 9. write to final table
		IF     iWafer_Level = 2 THEN      set stErrorLog = '710'; -- need wafer data 
            
            delete from ker_wip_w2_bt where lot_id_p1 = stLot_Id;           
            insert into ker_wip_w2_bt select * from ker_wip_w2_tt;
            
                                          set stErrorLog = '719';             
		ELSEIF iWafer_Level = 0 THEN      set stErrorLog = '720';-- normal omi

            delete from ker_wip_w0_bt where lot_id_p1 = stLot_Id;           
            insert into ker_wip_w0_bt select * from ker_wip_w0_tt;

                                          set stErrorLog = '729';
        -- ELSE 
        END IF;                           set stErrorLog = '799';
    
    
														set stErrorLog = '900';  
                                                         -- log execution time
                                                         SELECT 1 into iResult_1_ok_0_ng; -- return procedure ok or not
                                                         SELECT iResult_1_ok_0_ng; -- return value, let delphi code can receive it 2017/11/27 lkchena                                                                      
                                                               call sys_job_exec_sp( 9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog);-- log end:  -- iJob_Status: 1: under execute 2:compelete job -3: over spec -9:error 
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `o1fp_sorter_04_sub_32_posting_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `o1fp_sorter_04_sub_32_posting_sp`(IN stPType varchar(64),IN stCType varchar(64), IN stLot_Id varchar(64),IN stS varchar(64),IN stPart_Id varchar(64),IN stRoute_Id varchar(64),IN stOpe_No varchar(64),IN stUser varchar(64))
    COMMENT 'track in cart porcedure - 2020/03/01 lkchena'
label_sp: 
BEGIN
-- tip: o1fp_sorter_04_sub_32_posting_sp for forming station, others use order_trackin_cart_sp2 -- 2020/03/02 lkchena

-- test case: 2020/03/01 lkchena
-- call o1fp_sorter_04_sub_32_posting_sp('3','2','A0300106.0','R','ZZZ-02','ROUTE-ZZZ-01','200.220','user');
-- call o1fp_sorter_04_sub_32_posting_sp('5','2','A0300106.0','R','ZZZ-02','ROUTE-ZZZ-01','200.250','user');
-- call o1fp_sorter_04_sub_32_posting_sp('2','2','A0300106.0','R','ZZZ-02','ROUTE-ZZZ-01','200.105','user');
-- SELECT * FROM sys_job_log_bt ORDER BY time_start desc

-- zero record -- 2020/03/04 lkchena
-- delete from ker_wip_bt 
-- -- select * from ker_wip_bt 
-- where tool_id = 'AF35Y01'
 
  DECLARE stJob_Name varchar(32) DEFAULT "o1fp_sorter_04_sub_32_posting_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  -- DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  -- DECLARE stMfg_No varchar(16) DEFAULT NULL;
  -- DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  -- lot parameter  2020/03/01 lkchena
  DECLARE stHeader  varchar(1) DEFAULT NULL;
  DECLARE stNode_Id varchar(2) DEFAULT NULL; -- 00 ~ zz
  DECLARE iLot_Len int DEFAULT 0;
  DECLARE iLot_Cum int DEFAULT 0;
  
  -- 
  DECLARE stOpe_No2  varchar(16) DEFAULT NULL;
  DECLARE stS_2  varchar(1) DEFAULT NULL;-- default is 'Q'
  DECLARE stLot_Note  varchar(64) DEFAULT NULL;
  
  -- DECLARE stLot_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  DECLARE stCart_Id  varchar(16) DEFAULT NULL; -- 2020/03/02 lkchena 
  DECLARE iLot_Size  int DEFAULT 0;  
  DECLARE stRfid_Tag varchar(16) DEFAULT NULL;  

  -- dm use
  DECLARE iRD_Flag      int DEFAULT 0;
  -- DECLARE stOpe_No      varchar(16) DEFAULT NULL;
  -- DECLARE stPart_Id     varchar(24) DEFAULT NULL;
  DECLARE stMold_Id     varchar(24) DEFAULT NULL;
  DECLARE stPowder_Type varchar(16) DEFAULT NULL;
  DECLARE stPowder_Id   varchar(24) DEFAULT NULL;
  DECLARE stSTB_Plan    varchar(19) DEFAULT NULL;
  DECLARE stOrder_No    varchar(24) DEFAULT NULL; -- ? ORDER
  
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime; 
  
  DECLARE stRemain TEXT; -- array use
  DECLARE spilter CHAR(1);
  DECLARE pos9 INT DEFAULT 1 ;
  DECLARE str9 VARCHAR(1000);

  DECLARE doneCursor int DEFAULT 0;
/*
  DECLARE cur1 CURSOR FOR
  SELECT a.mfg_no, a.mold_id, a.tool_id
  FROM erp_prod_stb_t9_prod a
    INNER JOIN tool_order_bt b
      ON a.mfg_no = b.mfg_no
      AND a.mold_id = b.mold_id
      AND b.status1 <> 0
      and b.rd_flag = 0 and a.rd_flag = 0; 
*/
                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';
  -- 0.0 ini
   set stS_2 = 'Q';-- default is 'Q'
   set stLot_Note = '';
   
  -- 1.0 check if has next step
   if      stPType = '3' THEN -- next step

      select count(*) into iTmp   from dm_route_bt 
      where 1=1 -- and part_id = 'ZZZ-02'
        and route_id = stRoute_Id -- 'ROUTE-ZZZ-01'
        and ope_no > stOpe_No; -- '200.220';
     
      set stLot_Note = 'act: next step'; 
     
   ELSEIF stPType = '5' THEN -- previous step
  
      select count(*) into iTmp   from dm_route_bt 
      where 1=1 -- and part_id = 'ZZZ-02'
        and route_id = stRoute_Id -- 'ROUTE-ZZZ-01'
        and ope_no < stOpe_No; -- '200.220';

      set stLot_Note = 'act: previous step'; 
      
   ELSEIF stPType = '2' THEN -- assigned ope_no
  
      select count(*) into iTmp   from dm_route_bt 
      where 1=1 -- and part_id = 'ZZZ-02'
        and route_id = stRoute_Id -- 'ROUTE-ZZZ-01'
        and ope_no < stOpe_No; -- '200.105';

      set stLot_Note = 'act: assigned step'; 
      
   END IF;   
  
  -- 2.0 
    if iTmp <> 0 then -- normal has next step
    -- ---------------------------------------------------------------------------------------
                                                                        SET stErrorLog = '200';  
     -- 3.1 get next ope_no
	  if      stPType = '3' THEN

          select ope_no into stOpe_No2  from dm_route_bt 
          where 1=1 -- and part_id = 'ZZZ-02'
           and route_id = stRoute_Id -- 'ROUTE-ZZZ-0'
           and ope_no   > stOpe_No -- '200.220';
           order by ope_no asc
          limit 1; 
     
       ELSEIF stPType = '5' THEN
  
          select ope_no into stOpe_No2  from dm_route_bt 
          where 1=1 -- and part_id = 'ZZZ-02'
           and route_id = stRoute_Id -- 'ROUTE-ZZZ-0'
           and ope_no   < stOpe_No -- '200.220';
           order by ope_no desc
          limit 1; 
       
       ELSEIF stPType = '2' THEN
  
          set stOpe_No2 = stOpe_No;
       
       END IF;   

                                                                        SET stErrorLog = '202';  
      -- 3.2 insert ker_wip_bt
       replace into ker_wip_bt -- insert into ker_wip_bt
         ( report_time,cate,lot_id,lot_id_p,cart_id,lot_size,lot_size_spec,box_pcs_spec,box_spec,box_cnt,s,pri,
           part_id,part_raw,raw_2d_code,ope_cate,extra_step,claim_time,claim_user,ws_func,tool_func,er,
           pos_id,pos_desc,tool_id,tag_track_io,order_no,mfg_no,rd_flag,b_vendor_id,b_vendor_name,
           lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,stb_plan,stb_real,powder_id,powder_type,
           full_num,full_code,full_time,
           route_id,mold_id,cust_id,cust_name,ope_no,ope_name,stage_id,stage_name,stage_order,tool_grp_id,tool_grp,ws_type,tool_type1,tool_type2,tool_type3,in_out,area_id,area_name,proc_time
          ) -- ,cast_pcs_spec
       select 
           a.report_time,a.cate,a.lot_id,a.lot_id_p,a.cart_id,a.lot_size,a.lot_size_spec,a.box_pcs_spec,a.box_spec,a.box_cnt,stS_2,a.pri,-- a.s
           a.part_id,a.part_raw,a.raw_2d_code,a.ope_cate,a.extra_step,now(),stUser,a.ws_func,a.tool_func,a.er,-- a.claim_time,
           a.pos_id,a.pos_desc,a.tool_id,a.tag_track_io,a.order_no,a.mfg_no,a.rd_flag,a.b_vendor_id,a.b_vendor_name,
           stLot_Note,a.lot_memo,a.track_in_time,a.track_out_time,a.proc_start_time,a.proc_end_time,a.stb_plan,a.stb_real,a.powder_id,a.powder_type, -- a.lot_note
           a.full_num,a.full_code,a.full_time,
           b.route_id,b.mold_id,b.cust_id,b.cust_name,b.ope_no,b.ope_name,b.stage_id,b.stage_name,b.stage_order,b.tool_grp_id,b.tool_grp,b.ws_type,b.tool_type1,b.tool_type2,b.tool_type3,b.in_out,b.area_id,b.area_name,b.proc_time 
           -- ,b.cast_pcs_spec
        from ker_wip_bt a  cross join ( select * from dm_route_bt 
		       							 where route_id = stRoute_Id -- 'ROUTE-ZZZ-01'
                                             and  ope_no = stOpe_No2 -- '200.250'
              ) b 
           where a.lot_id = stLot_Id; -- 'A0300106.0'
      
      
      -- 3.2 get lot infor
--       select   rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   stb_plan,   order_no,   cart_id,  lot_size into
--	    	   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No, stCart_Id, iLot_Size
--          from ker_wip_bt
--       where lot_id = stLot_Id;                                         SET stErrorLog = '204';

      -- 3.3 
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
    -- ---------------------------------------------------------------------------------------
    else -- complete all steps    
    -- ---------------------------------------------------------------------------------------
                                                                        SET stErrorLog = '800';  
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
                                                       
    -- ---------------------------------------------------------------------------------------
	end if;
  
  
/*
  -- 1. get lot parameter: 
   SELECT value1 as header,value2 as node_id,ivalue1 as lot_len,ivalue2 as lot_cum 
         into stHeader,stNode_Id,iLot_Len,iLot_Cum
   FROM sys_param_conf_bt where param_id = 'LOT-01'; SET stErrorLog = '001';
   
                               IF stHeader is NULL THEN -- no setting 
                                      SET stErrorLog = 'no sys_param_conf_bt data'; signal sqlstate '45000' set message_text = stErrorLog;
                                      leave label_sp;     
							   END IF;
   -- 1.2 get order base info 
    select  rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   date_stb,   order_no into
		   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No
      from tool_order_bt 
    where 1=1
     and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
     and mfg_no = stMfg_No;                           SET stErrorLog = '003';

   -- 1.8 ker variant
   SET dREPORT_TIME9 = NOW();  
   
   -- qqq: user is not full ready, so we use command part to replace it   
   -- 1.9 use command part
   set stPart_Id = 'ZZZ-01';
   set stMold_Id = 'ZZZ-01';
            
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;

WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '010';
	  -- 3.1.0
      set stRfid_Tag = str9;
   
      -- 3.1.1 add lot cum and get lot_id
      set iLot_Cum = iLot_Cum + 1;
      set stTmp = LPAD( iLot_Cum, iLot_Len, '0' );
      set stLot_Id = concat(stHeader,stNode_Id,stTmp); -- test:  select stLot_Id;
   
      -- 3.1.2 get cart_id, lot_size / '5060041'
      select a.cart_id, b.cnt_total into stCart_Id, iLot_Size   -- a.cart_type, , b.cnt_box
         from mcs_cart_base_bt a left join mcs_cart_type_bt b on a.cart_type = b.cart_type
       where rfid_tag = stRfid_Tag and active = 1;     SET stErrorLog = '010';

      -- 5.1 insert ker
			 INSERT INTO ker_wip_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, -- 
                                      cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id,claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                      part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                                      ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                                      )
                  SELECT dREPORT_TIME9, 'RT',stLot_Id,stLot_Id,'R' AS s, iLot_Size, iLot_Size, 350, -- 
                        stCart_Id,stTool_Id, stOrder_No, stMfg_No, stSTB_Plan, stPowder_Type, stPowder_Id, now(), now(),-- iLOT_SIZE_SPEC9,iCAST_SPEC9,iCAST_PCS_SPEC9,
                        part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                        ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                   FROM dm_flow_bt 
                  WHERE part_id = stPart_Id -- 'ZZZ-01' / '100.00'
                    AND  ope_no = stOpe_No;            SET stErrorLog = '011';
   
	  -- FOR DEBUG: SELECT  concat(stPart_Id,' - ',  stLot_Id,' - ', stRfid_Tag,' - ',stCart_Id,' - ',iLot_Size,' - ',stPowder_Type );

                                                  SET stErrorLog = '020';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '091';

   -- 6.1 update too_order_bt.status1 = 6 ???
   --     seems not to update is ok, if update to 6, then track-out need rollback to 5 ...
   --     ....
      update tool_order_bt 
        set status1 = 6
      where 1=1
       and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
       and mfg_no = stMfg_No; 
     
   -- 6.2 update lot cum number
	  UPDATE sys_param_conf_bt
         set ivalue2 = iLot_Cum -- + 1, add above
      WHERE param_id = 'LOT-01';                  SET stErrorLog = '092';
 
   -- 6.1 update tool status
    UPDATE oee_tool_bt set status = 'RUN'  WHERE tool_id = stTool_id;



                                                  SET stErrorLog = '099';
*/
/*  
  OPEN cur1;
  REPEAT
    FETCH cur1 INTO stMfg_No, stMold_Id, stTool_Id;
    IF NOT doneCursor THEN
      SET stErrorLog = CONCAT('030_', stMfg_No);
      
      DELETE FROM erp_prod_stb_t9_prod
      WHERE mfg_no = stMfg_No
        AND mold_id = stMold_Id 
        and rd_flag = 0;
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CLOSE cur1;
  SET stErrorLog = CONCAT('061_', 'insert compact table');

  
  DELETE FROM tool_order_bt
  WHERE (mfg_no, mold_id) IN (SELECT mfg_no,mold_id FROM erp_prod_stb_t9_prod)
    and rd_flag = 0; 

  SET stErrorLog = '070';

  
  INSERT INTO tool_order_bt (pri, status1, tool_grp_id, tool_id, part_id, part_raw, part_name, part_erp_no, mold_id, powder_type, powder_erp_no, cnt_plan, cnt_act, date_due, date_stb, date_comp, mfg_no, order_no, rd_flag, cust_id, weight, unit, owner_id, owner_name, note, rec_user, rec_time,
                             time_order, ope_no, ope_name, stage_id, stage_name, stage_order) 
    SELECT pri,status1,tool_grp_id,tool_id,part_id,part_raw,part_name,part_erp_no,mold_id,powder_type,powder_erp_no,cnt_plan,cnt_act,date_due,date_stb,date_comp,mfg_no,order_no,rd_flag,cust_id,weight,unit,owner_id,owner_name,note,rec_user,rec_time,
           DATE_FORMAT(NOW(),"%Y/%m/%d %H:00:00") AS time_order,'100.00','成型作業','Forming','成型站', '100' 
          
    FROM erp_prod_stb_t9_prod
  where 1=1
    and rd_flag = 0; 
  
  
*/  

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `oee_sum_daily_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `oee_sum_daily_sp`(
	IN `dDateTimeIn` datetime
,
	IN `iType` INT

)
BEGIN













  declare  stJob_Name varchar(32) DEFAULT 'oee_sum_daily_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stError varchar(64);
  DECLARE  dReportTime1,dReportTime2 datetime;
      
                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
                                                      SELECT 0 into iResult_1_ok_0_ng; 
                                                  ROLLBACK;     
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stError); 
                                                      SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              set stError = '000';    
 
 
 IF iType = 1 THEN	
 
  select date_add(DATE(dDateTimeIn), interval 0 day),date_add(DATE(dDateTimeIn), interval +1 day)
    INTO dReportTime1,dReportTime2;                                               set stError = '010';
    
 ELSEIF iType = 2 THEN
 																											set stError = '010-1';
  select date_add(DATE(dDateTimeIn), interval -1 day),date_add(DATE(dDateTimeIn), interval 0 day)
    INTO dReportTime1,dReportTime2; 
	 
 END IF;	  
                                                                                  set stError = '010-2';
 
 

   DELETE FROM oee_sum_tt;                                     set stError = '011';  

   delete from oee_sum_st where report_time = dReportTime1;    set stError = '011-1'; 
   
 
   INSERT INTO oee_sum_tt
    SELECT dReportTime1,'YES',a.tool_id, a.device, 
	         avg(avl),avg(eff),avg(lost),avg(down),avg(pm),avg(hold),avg(setup),avg(test),avg(mon),avg(wait),avg(off),
      	   sum(move_act),sum(move_prod),sum(move_test),sum(move_eng),sum(move_avg)       
      FROM mes_dev.oee_ss_hr_bt a
    where 1=1
      and a.report_time >= dReportTime1 
      and a.report_time <  dReportTime2 
    group by a.tool_id, a.device;
                                                                set stError = '012';     
   
  delete from oee_sum_st where report_time = dReportTime1;
  insert into oee_sum_st select * from oee_sum_tt;  
                                                                set stError = '019';     

 
 
  DELETE FROM oee_tool_sch_bth  WHERE claim_time >= dReportTime1;          set stError = '711';  

 
  INSERT INTO oee_tool_sch_bth
  SELECT * FROM oee_tool_sch_bt WHERE claim_time <=  dReportTime1;         set stError = '712';

 
  DELETE FROM oee_tool_sch_bt   WHERE claim_time <=  dReportTime1;         set stError = '713';

                                                              set stError = '900'; 
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `oee_sum_hourly_cur1_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `oee_sum_hourly_cur1_sp`(
	IN `dReportTime1` datetime,
	IN `dReportTime2` datetime




)
BEGIN










  declare  stJob_Name varchar(32) DEFAULT 'oee_sum_hourly_cur1_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stError varchar(64);

  DECLARE  itype int DEFAULT 0;
  DECLARE  itemp int DEFAULT 0;
                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
                                                      SELECT 0 into iResult_1_ok_0_ng; 
                                                      ROLLBACK;     
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stError); 
                                                      SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              set stError = '000';   
    
    DROP TEMPORARY TABLE IF EXISTS oee_sum_hourly_cur1_t0;
    CREATE TEMPORARY TABLE oee_sum_hourly_cur1_t0(
        tool_id varchar(12),
        status  varchar(6),
        value   float(5,2)
    );



    
    insert into oee_sum_hourly_cur1_t0
     SELECT a.tool_id, a.status, 
               SUM(hr)*100/( (UNIX_TIMESTAMP(dReportTime2) - UNIX_TIMESTAMP(dReportTime1))/60/60 ) AS percent
        FROM ( SELECT a.tool_id, a.status, a.time_start, a.time_end,      
                      (UNIX_TIMESTAMP(a.time_end) - UNIX_TIMESTAMP(a.time_start))/60/60 AS hr               
                  FROM ( 
                    SELECT a.tool_id, a.status, a.claim_time AS time_start, 
                                 IFNULL( lead(DATE_FORMAT(a.claim_time, "%Y/%m/%d %H:%i:%S"),1) over
                                    (partition by a.tool_id order by DATE_FORMAT(a.claim_time, "%Y/%m/%d %H:%i:%S") asc), dReportTime2 ) as 'time_end'
                            FROM oee_tool_sch_uvh a
                          WHERE 1=1
                            AND a.device = 'MAIN' 
                            and a.claim_time >= dReportTime1 
                            and a.claim_time <  dReportTime2 
                         union all		
	      		 			    			 SELECT a.tool_id, a.status,  dReportTime1  AS time_start , 
                        		IFNULL( c.claim_time , dReportTime2 ) as 'time_end'
                        		FROM oee_tool_sch_uvh a inner join                        		
                              	(
											          	SELECT distinct b.tool_id, b.status, b.claim_time
                              		FROM oee_tool_sch_uvh b
                                    WHERE 1=1
                                    AND b.device = 'MAIN' 
                                    AND b.claim_time >= dReportTime1
                                    AND b.claim_time <  dReportTime2                                  
                                 ) as c
                        	        	on a.tool_id = c.tool_id
                        	         	WHERE 1=1
                        		        AND a.device = 'MAIN' 
                                    and a.claim_time < dReportTime1
                        		        group by a.tool_id desc	               
                      ) a
          ) a
         group BY a.tool_id, a.status; 


       SELECT COUNT(distinct a.tool_id) FROM oee_sum_hourly_cur1_t0 a  INTO itype;

       SELECT COUNT(distinct a.tool_id) FROM oee_tool_sch_uvh a where a.claim_time < dReportTime1 INTO itemp;
            
 
       IF itype = 0 THEN
       
            insert into oee_sum_hourly_cur1_t0
            SELECT a.tool_id, a.status,'100' AS percent 
            FROM oee_tool_sch_uvh a  
            WHERE 1=1
            AND a.device = 'MAIN' 
            and a.claim_time < dReportTime1
            group by a.tool_id DESC;    
           
       ELSEIF itype = itemp THEN  
		     
           SET itemp = 0;    
           
       ELSE
       
           insert into oee_sum_hourly_cur1_t0            
            SELECT distinct b.* from oee_sum_hourly_cur1_t0 a inner join 
			   (  SELECT a.tool_id, a.status,'100' AS percent 
            FROM oee_tool_sch_uvh a  
            WHERE 1=1
            AND a.device = 'MAIN' 
            and a.claim_time < dReportTime1
            group by a.tool_id DESC ) b
            on a.tool_id != b.tool_id; 
        
        END IF;



                                                              select '900' INTO stError;
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');



END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `oee_sum_hourly_s01_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `oee_sum_hourly_s01_sp`(
   IN dReportTime1 datetime, in dReportTime2 datetime
)
BEGIN








  declare  stJob_Name varchar(32) DEFAULT 'oee_sum_hourly_s01_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stError varchar(64);

  declare  stTool_id,stTool_Tmp varchar(12) default '';
  declare  eMove_Prod,eMove_Test,eMove_Eng,eMove_Avg float DEFAULT 0;

  declare doneCursor INT DEFAULT 0;
  declare cur1 CURSOR FOR 
               select tool_id, sum(if(rd_flag=0,move,0)) move_prod, 
                               sum(if(rd_flag=1,move,0)) move_test, 
                               sum(if(rd_flag=2,move,0)) move_eng 
                from ( SELECT a.tool_id, a.rd_flag, count(*) as move
                    FROM dcop_collect_bt a
                   where 1=1
                    and a.claim_time >= dReportTime1 
                    and a.claim_time <  dReportTime2 
                    group by a.tool_id, a.rd_flag
                  ) a1
                group by tool_id; 
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;

                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
                                                      SELECT 0 into iResult_1_ok_0_ng; 
                                                      ROLLBACK;     
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stError); 
                                                      SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              set stError = '000';    

  
   
   set eMove_Avg = 1*60*60/5;

  
    OPEN cur1;
    REPEAT
      FETCH cur1 INTO stTool_id,eMove_Prod,eMove_Test,eMove_Eng;
      IF NOT doneCursor THEN                         set stError = '010';       





          update oee_ss_hr_tt 
            set move_prod = eMove_Prod, move_test = eMove_Test, move_eng = eMove_Eng, Move_Avg = eMove_Avg,
                eff = (eMove_Prod / eMove_Avg)*100 
            
          where 1=1
            and tool_id = stTool_id;     
                                                     set stError = '020'; 
      END IF;
    UNTIL doneCursor END REPEAT;
    CLOSE cur1;

                                                              select '900' INTO stError;
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');


END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `oee_sum_hourly_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `oee_sum_hourly_sp`(
	IN `dDateTimeIn` datetime


)
BEGIN









  declare  stJob_Name varchar(32) DEFAULT 'oee_sum_hourly_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stError varchar(64);

  declare  dReportTime1,dReportTime2 datetime; 
  declare  stTool_id,stTool_Tmp varchar(12) default '';
  declare  stStatus  varchar(6) default '';
  declare  eValue float(5,2) DEFAULT 0;
  declare  eAvl,eEff,eLost,eDown,ePm,eHold,eSetup,eTest,eMon,eWait,eOff float(5,2) DEFAULT 0;
  declare  eMove_Act,eMove_Prod,eMove_Test,eMove_Eng,eMove_Avg float DEFAULT 0;

  declare doneCursor INT DEFAULT 0;
  
  declare cur1 CURSOR FOR 
  SELECT a.tool_id, a.status, a.value  FROM oee_sum_hourly_cur1_t0 a; 
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;

                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
                                                      SELECT 0 into iResult_1_ok_0_ng; 
                                                      ROLLBACK;     
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stError); 
                                                      SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              set stError = '000';    


 
 
 
 
  SELECT DATE_FORMAT(dDateTimeIn, "%Y/%m/%d %H:00:00"), DATE_ADD( DATE_FORMAT(dDateTimeIn, "%Y/%m/%d %H:00:00"), INTERVAL 1 HOUR )
    into dReportTime1, dReportTime2;                           set stError = '001';    

 
  delete from oee_ss_hr_tt;

  
   CALL oee_sum_hourly_cur1_sp(dReportTime1,dReportTime2);
   


  
    OPEN cur1;
    REPEAT
      FETCH cur1 INTO stTool_id,stStatus,eValue;
      IF NOT doneCursor THEN   
                                                    set stError = '010';     
                                                                                                 
          
          if ( stTool_Id <> stTool_Tmp ) and ( stTool_Tmp <> '' ) THEN set stError = '011'; 
          
              INSERT INTO oee_ss_hr_tt ( report_time,cate,tool_id,device,                
                                        avl,eff,lost,down,pm,hold,setup,test,mon,wait,off,
                                        move_act,move_prod,move_eng,move_avg
              ) VALUES ( dReportTime1,'RT',stTool_Tmp,'MAIN',
                        eAvl,eEff,eLost,eDown,ePm,eHold,eSetup,eTest,eMon,eWait,eOff,
                        eMove_Act,eMove_Prod,eMove_Eng,eMove_Avg
              );                                    set stError = '019'; 

           SET eAvl   = 0;   SET eLost  = 0;  SET eDown   = 0;  SET ePm    = 0;  SET eHold   = 0;  SET eSetup   = 0;
           SET eTest  = 0;   SET eMon   = 0;  SET eWait   = 0;  SET eOff   = 0;  SET eAvl   = 0;

          
          end if;
                                                    set stError = '020'; 
          
           set stTool_Tmp = stTool_Id;


           IF     STRCMP(stStatus,'UP' )   = 0 THEN SET eAvl   = eValue;
           ELSEIF STRCMP(stStatus,'LOST' ) = 0 THEN SET eLost  = eValue;
           ELSEIF STRCMP(stStatus,'DOWN' ) = 0 THEN SET eDown  = eValue;
           ELSEIF STRCMP(stStatus,'PM'   ) = 0 THEN SET ePm    = eValue;
           ELSEIF STRCMP(stStatus,'HOLD' ) = 0 THEN SET eHold  = eValue;
           ELSEIF STRCMP(stStatus,'SETUP') = 0 THEN SET eSetup = eValue;
           ELSEIF STRCMP(stStatus,'TEST' ) = 0 THEN SET eTest  = eValue;
           ELSEIF STRCMP(stStatus,'MON'  ) = 0 THEN SET eMon   = eValue;
           ELSEIF STRCMP(stStatus,'WAIT' ) = 0 THEN SET eWait  = eValue;
           ELSEIF STRCMP(stStatus,'OFF'  ) = 0 THEN SET eOff   = eValue;
           END IF;




      END IF;
    UNTIL doneCursor END REPEAT;
    CLOSE cur1;



   

                          if ( stTool_Tmp <> '' ) THEN set stError = '011b'; 
                             INSERT INTO oee_ss_hr_tt ( report_time,cate,tool_id,device,avl,eff,lost,down,pm,hold,setup,test,mon,wait,off,move_act,move_prod,move_eng,move_avg
                              ) VALUES ( dReportTime1,'RT',stTool_Tmp,'MAIN',eAvl,eEff,eLost,eDown,ePm,eHold,eSetup,eTest,eMon,eWait,eOff,eMove_Act,eMove_Prod,eMove_Eng,eMove_Avg
                             );                                    set stError = '019-b'; 
                          end if;



   
   
    CALL oee_sum_hourly_s01_sp(dReportTime1,dReportTime2);
    

   
   
   
   
   








   
  delete from oee_ss_hr_bt where report_time = dReportTime1;
  insert into oee_ss_hr_bt select * from oee_ss_hr_tt;



 

   

 
 
                                                              select '900' INTO stError;
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');


END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `om_trackin_cart_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `om_trackin_cart_sp`(IN stFlag varchar(1), IN stTool_Id varchar(12), IN stCart_No varchar(24), IN stLot_Id varchar(24), IN stUser varchar(24))
    COMMENT 'track in cart porcedure - 2020/03/01 lkchena'
label_sp: 
BEGIN
-- tip1: stFlag: 1: track-in 0:cancel track-in

-- test case: 2020/03/01 lkchena
/*
 call om_trackin_cart_sp('1','ASCNY06#B','ACS011','A0100104.0','SYS');
 call om_trackin_cart_sp('1','ASCNY06#B','ACS011','A0100103.0','SYS');
 call om_trackin_cart_sp('0','ASCNY06#B','ACS011','A0100103.0','SYS');
 select * from ker_wip_bt order by report_time desc
 select * from oee_tool_bt where tool_id = 'ASCNY06#B'
 SELECT * FROM sys_job_log_bt ORDER BY time_start desc
 
-- zero record -- 2020/03/04 lkchena
 update ker_wip_bt set s='Q',cart_no = '', tool_id = '', claim_user = ''
 where ws_type = 21 and tool_type1 = 21;
*/

  DECLARE stJob_Name varchar(32) DEFAULT "om_trackin_cart_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  
  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  -- DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  -- DECLARE stMfg_No varchar(16) DEFAULT NULL;
  -- DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  -- lot parameter  2020/03/01 lkchena
  DECLARE stHeader  varchar(1) DEFAULT NULL;
  DECLARE stNode_Id varchar(2) DEFAULT NULL; -- 00 ~ zz
  DECLARE iLot_Len int DEFAULT 0;
  DECLARE iLot_Cum int DEFAULT 0;
  
  -- D0ECLARE stLot_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  -- DECLARE stCart_No  varchar(16) DEFAULT NULL; -- 2020/03/02 lkchena 
  DECLARE iLot_Size  int DEFAULT 0;  
  DECLARE stRfid_Tag varchar(16) DEFAULT NULL;  

  -- dm use
  DECLARE iRD_Flag      int DEFAULT 0;
  DECLARE stOpe_No      varchar(16) DEFAULT NULL;
  DECLARE stPart_Id     varchar(24) DEFAULT NULL;
  DECLARE stMold_Id     varchar(24) DEFAULT NULL;
  DECLARE stPowder_Type varchar(16) DEFAULT NULL;
  DECLARE stPowder_Id   varchar(24) DEFAULT NULL;
  DECLARE stSTB_Plan    varchar(19) DEFAULT NULL;
  DECLARE stOrder_No    varchar(24) DEFAULT NULL; -- ? ORDER
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime;   
  DECLARE doneCursor int DEFAULT 0;
                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '000');
                                               SET stErrorLog = '000';
  -- 1: track-in 0: cancel track-in
  if stFlag = '1' then  SET stErrorLog = '100';
  -- -------------------------------------------------------------------------------- 
	 -- 1. update ker wip 
      update ker_wip_bt 
         set s='R',cart_no = stCart_No, tool_id = stTool_Id, claim_user = stUser,
             report_time = now(), claim_time = now(), track_in_time = now(), 
             lot_note = 'act: track-in'
       where 1=1
        and lot_id = stLot_Id;   SET stErrorLog = '110';
     
     -- 2. update oee
      update oee_tool_bt x
        INNER JOIN ker_wip_bt a on a.lot_id = stLot_Id -- 'A0100103.0'
       set x.status = 'RUN', 
           x.mfg_no = a.mfg_no,
           x.part_id = a.part_id,
           x.route_id = a.route_id,
           x.rd_flag = a.rd_flag,
           x.stage_id = a.stage_id,
           x.stage_name = a.stage_name,
           x.stage_order = a.stage_order,
           x.ope_no = a.ope_no,
           x.ope_name = a.ope_name,
           x.cust_id = a.cust_id,
           x.cust_name = a.cust_name
       where x.tool_id = stTool_Id; -- 'ASCNY06#B'    
     
  -- -------------------------------------------------------------------------------- 
  ELSEIF stFlag = '0' THEN SET stErrorLog = '200';
  -- -------------------------------------------------------------------------------- 
  
	 -- 1. update ker wip 
      update ker_wip_bt 
         set s='Q',cart_no = '', tool_id = '', claim_user = stUser,
             report_time = now(), claim_time = now(), track_in_time = null, 
             lot_note = 'act: cancel track-in'
       where 1=1
        and lot_id = stLot_Id;   SET stErrorLog = '210';
   
     -- 2.1 check if has others lot still run
       select count(*) into iTmp from ker_wip_bt
       where tool_id = stTool_Id -- 'ASCNY06#B' 
        and s = 'R';  SET stErrorLog = '220';   
     
     -- 2.2 oee table
       if iTmp = 0  then  
         
          update oee_tool_bt x
           set x.status = 'LOST',
               x.mfg_no = NULL,
               x.part_id = NULL,
               x.route_id = NULL,
               x.rd_flag = NULL,
               x.stage_id = NULL,
               x.stage_name = NULL,
               x.stage_order = NULL,
               x.ope_no = NULL,
               x.ope_name = NULL,
               x.cust_id = NULL,
               x.cust_name = NULL
           where x.tool_id = stTool_Id; -- 'ASCNY06#B'
       
       end if;  SET stErrorLog = '230';
 
 
 
 -- select * from oee_tool_bt where tool_id = 'ASCNY06#B'     
   
   
  -- -------------------------------------------------------------------------------- 
  else  SET stErrorLog = '300';
  -- -------------------------------------------------------------------------------- 
  
   
  -- -------------------------------------------------------------------------------- 
  end if;

  
  
/*
  -- 1. get lot parameter: 
   SELECT value1 as header,value2 as node_id,ivalue1 as lot_len,ivalue2 as lot_cum 
         into stHeader,stNode_Id,iLot_Len,iLot_Cum
   FROM sys_param_conf_bt where param_id = 'LOT-01'; SET stErrorLog = '001';
   
                               IF stHeader is NULL THEN -- no setting 
                                      SET stErrorLog = 'no sys_param_conf_bt data'; signal sqlstate '45000' set message_text = stErrorLog;
                                      leave label_sp;     
							   END IF;
   -- 1.2 get order base info 
    select  rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   date_stb,   order_no into
		   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No
      from tool_order_bt 
    where 1=1
     and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
     and mfg_no = stMfg_No;                           SET stErrorLog = '003';

   -- 1.8 ker variant
   SET dREPORT_TIME9 = NOW();  
   
   -- qqq: user is not full ready, so we use command part to replace it   
   -- 1.9 use command part
   set stPart_Id = 'ZZZ-01';
   set stMold_Id = 'ZZZ-01';
            
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;

WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '010';
	  -- 3.1.0
      set stRfid_Tag = str9;
   
      -- 3.1.1 add lot cum and get lot_id
      set iLot_Cum = iLot_Cum + 1;
      set stTmp = LPAD( iLot_Cum, iLot_Len, '0' );
      set stLot_Id = concat(stHeader,stNode_Id,stTmp); -- test:  select stLot_Id;
   
      -- 3.1.2 get cart_id, lot_size / '5060041'
      select a.cart_id, b.cnt_total into stCart_No, iLot_Size   -- a.cart_type, , b.cnt_box
         from mcs_cart_base_bt a left join mcs_cart_type_bt b on a.cart_type = b.cart_type
       where rfid_tag = stRfid_Tag and active = 1;     SET stErrorLog = '010';

      -- 5.1 insert ker
			 INSERT INTO ker_wip_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, -- 
                                      cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id,claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                      part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                                      ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                                      )
                  SELECT dREPORT_TIME9, 'RT',stLot_Id,stLot_Id,'R' AS s, iLot_Size, iLot_Size, 350, -- 
                        stCart_No,stTool_Id, stOrder_No, stMfg_No, stSTB_Plan, stPowder_Type, stPowder_Id, now(), now(),-- iLOT_SIZE_SPEC9,iCAST_SPEC9,iCAST_PCS_SPEC9,
                        part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                        ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                   FROM dm_flow_bt 
                  WHERE part_id = stPart_Id -- 'ZZZ-01' / '100.00'
                    AND  ope_no = stOpe_No;            SET stErrorLog = '011';
   
	  -- FOR DEBUG: SELECT  concat(stPart_Id,' - ',  stLot_Id,' - ', stRfid_Tag,' - ',stCart_No,' - ',iLot_Size,' - ',stPowder_Type );

                                                  SET stErrorLog = '020';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '091';

   -- 6.1 update too_order_bt.status1 = 6 ???
   --     seems not to update is ok, if update to 6, then track-out need rollback to 5 ...
   --     ....
      update tool_order_bt 
        set status1 = 6
      where 1=1
       and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
       and mfg_no = stMfg_No; 
     
   -- 6.2 update lot cum number
	  UPDATE sys_param_conf_bt
         set ivalue2 = iLot_Cum -- + 1, add above
      WHERE param_id = 'LOT-01';                  SET stErrorLog = '092';
 
   -- 6.1 update tool status
    UPDATE oee_tool_bt set status = 'RUN'  WHERE tool_id = stTool_id;



                                                  SET stErrorLog = '099';

/*  
  OPEN cur1;
  REPEAT
    FETCH cur1 INTO stMfg_No, stMold_Id, stTool_Id;
    IF NOT doneCursor THEN
      SET stErrorLog = CONCAT('030_', stMfg_No);
      
      DELETE FROM erp_prod_stb_t9_prod
      WHERE mfg_no = stMfg_No
        AND mold_id = stMold_Id 
        and rd_flag = 0;
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CLOSE cur1;
  SET stErrorLog = CONCAT('061_', 'insert compact table');

  
  DELETE FROM tool_order_bt
  WHERE (mfg_no, mold_id) IN (SELECT mfg_no,mold_id FROM erp_prod_stb_t9_prod)
    and rd_flag = 0; 

  SET stErrorLog = '070';

  
  INSERT INTO tool_order_bt (pri, status1, tool_grp_id, tool_id, part_id, part_raw, part_name, part_erp_no, mold_id, powder_type, powder_erp_no, cnt_plan, cnt_act, date_due, date_stb, date_comp, mfg_no, order_no, rd_flag, cust_id, weight, unit, owner_id, owner_name, note, rec_user, rec_time,
                             time_order, ope_no, ope_name, stage_id, stage_name, stage_order) 
    SELECT pri,status1,tool_grp_id,tool_id,part_id,part_raw,part_name,part_erp_no,mold_id,powder_type,powder_erp_no,cnt_plan,cnt_act,date_due,date_stb,date_comp,mfg_no,order_no,rd_flag,cust_id,weight,unit,owner_id,owner_name,note,rec_user,rec_time,
           DATE_FORMAT(NOW(),"%Y/%m/%d %H:00:00") AS time_order,'100.00','成型作業','Forming','成型站', '100' 
          
    FROM erp_prod_stb_t9_prod
  where 1=1
    and rd_flag = 0; 
  
  
*/  

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog);

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `order_trackin_box_nor_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `order_trackin_box_nor_sp`(stTool_Id VARCHAR(64),stMFG_NO VARCHAR(64),stLot_Id varchar(64),stArray varchar(255),
   iLot_Size int,iNum_Default int,iNum_Last int, iUnit_Type int)
    COMMENT 'track in box porcedure - 2020/05/05 lkchena'
label_sp: 
BEGIN
-- tip: order_trackin_box_nor_sp for forming station, others use order_trackin_cart_sp2 -- 2020/03/02 lkchena

-- test case: 2020/03/01 lkchena
/* 
 call order_trackin_box_nor_sp('ASCNY06#B','T-MFG_NO-003','A0100104.0','ACS00301',365,120,8,0);
 call order_trackin_box_nor_sp('ASCNY06#B','T-MFG_NO-003','A0100104.0','ACS00301,ACS00302,ACS00303,ACS00305',365,120,8,0);
 select * from ker_wip_w0_bt order by report_time desc
 SELECT * FROM sys_job_log_bt ORDER BY time_start desc
*/
-- zero record -- 2020/03/04 lkchena
-- delete from ker_wip_bt 
-- -- select * from ker_wip_bt 
-- where tool_id = 'AF35Y01'
 

  DECLARE stJob_Name varchar(32) DEFAULT "order_trackin_box_nor_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  
  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  DECLARE stBox_No  varchar(16) DEFAULT NULL;
  -- DECLARE iLot_Size2 int DEFAULT 0; -- we can't get existed lot size!!! final lot_size should different with lot_size original 2020/05/05 lkchena
  DECLARE iBox_Size  int DEFAULT 0;

  -- DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  -- DECLARE stMfg_No varchar(16) DEFAULT NULL;
  -- DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  -- lot parameter  2020/03/01 lkchena
  DECLARE stHeader  varchar(1) DEFAULT NULL;
  DECLARE stNode_Id varchar(2) DEFAULT NULL; -- 00 ~ zz
  DECLARE iLot_Len int DEFAULT 0;
  DECLARE iLot_Cum int DEFAULT 0;
  
  -- DECLARE stLot_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  DECLARE stCart_Id  varchar(16) DEFAULT NULL; -- 2020/03/02 lkchena 
  -- DECLARE iLot_Size  int DEFAULT 0;  
  DECLARE stRfid_Tag varchar(16) DEFAULT NULL;  

  -- dm use
  DECLARE iRD_Flag      int DEFAULT 0;
  DECLARE stOpe_No      varchar(16) DEFAULT NULL;
  DECLARE stPart_Id     varchar(24) DEFAULT NULL;
  DECLARE stMold_Id     varchar(24) DEFAULT NULL;
  DECLARE stPowder_Type varchar(16) DEFAULT NULL;
  DECLARE stPowder_Id   varchar(24) DEFAULT NULL;
  DECLARE stSTB_Plan    varchar(19) DEFAULT NULL;
  DECLARE stOrder_No    varchar(24) DEFAULT NULL; -- ? ORDER
  
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime; 
  
  DECLARE stRemain TEXT; -- array use
  DECLARE spilter CHAR(1);
  DECLARE pos9 INT DEFAULT 1 ;
  DECLARE str9 VARCHAR(1000);

  DECLARE doneCursor int DEFAULT 0;
                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';

/*
  -- 1. get lot parameter: 
   SELECT value1 as header,value2 as node_id,ivalue1 as lot_len,ivalue2 as lot_cum 
         into stHeader,stNode_Id,iLot_Len,iLot_Cum
   FROM sys_param_conf_bt where param_id = 'LOT-01'; SET stErrorLog = '001';
   
                               IF stHeader is NULL THEN -- no setting 
                                      SET stErrorLog = 'no sys_param_conf_bt data'; signal sqlstate '45000' set message_text = stErrorLog;
                                      leave label_sp;     
							   END IF;
   -- 1.2 get order base info 
    select  rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   date_stb,   order_no into
		   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No
      from tool_order_bt 
    where 1=1
     and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
     and mfg_no = stMfg_No;                           SET stErrorLog = '003';

   -- 1.8 ker variant
   SET dREPORT_TIME9 = NOW();  
   
   -- qqq: user is not full ready, so we use command part to replace it   
   -- 1.9 use command part
   set stPart_Id = 'ZZZ-01';
   set stMold_Id = 'ZZZ-01';
*/            
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;
  
WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '610';
	  -- 3.1.0
      set stBox_No  = str9;
      set iBox_Size = iNum_Default;
      if ( pos9 = 0 ) and ( iNum_Last <> 0 ) then set iBox_Size = iNum_Last; end if;
     
     select stBox_No, iLot_Size, iNum_Default, iNum_Last,iBox_Size;
     
/*      -- 3.1.1 add lot cum and get lot_id
      set iLot_Cum = iLot_Cum + 1;
      set stTmp = LPAD( iLot_Cum, iLot_Len, '0' );
      set stLot_Id = concat(stHeader,stNode_Id,stTmp); -- test:  select stLot_Id;
   
      -- 3.1.2 get cart_id, lot_size / '5060041'
      select a.cart_id, b.cnt_total into stCart_Id, iLot_Size   -- a.cart_type, , b.cnt_box
         from mcs_cart_base_bt a left join mcs_cart_type_bt b on a.cart_type = b.cart_type
       where rfid_tag = stRfid_Tag and active = 1;     SET stErrorLog = '010';

      -- 5.1 insert ker
			 INSERT INTO ker_wip_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, -- 
                                      cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id,claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                      part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                                      ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                                      )
                  SELECT dREPORT_TIME9, 'RT',stLot_Id,stLot_Id,'R' AS s, iLot_Size, iLot_Size, 350, -- 
                        stCart_Id,stTool_Id, stOrder_No, stMfg_No, stSTB_Plan, stPowder_Type, stPowder_Id, now(), now(),-- iLOT_SIZE_SPEC9,iCAST_SPEC9,iCAST_PCS_SPEC9,
                        part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                        ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                   FROM dm_flow_bt 
                  WHERE part_id = stPart_Id -- 'ZZZ-01' / '100.00'
                    AND  ope_no = stOpe_No;            SET stErrorLog = '011';
   
	  -- FOR DEBUG: SELECT  concat(stPart_Id,' - ',  stLot_Id,' - ', stRfid_Tag,' - ',stCart_Id,' - ',iLot_Size,' - ',stPowder_Type );
*/
                                                  SET stErrorLog = '620';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '691';

/*
   -- 6.1 update too_order_bt.status1 = 6 ???
   --     seems not to update is ok, if update to 6, then track-out need rollback to 5 ...
   --     ....
      update tool_order_bt 
        set status1 = 6
      where 1=1
       and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
       and mfg_no = stMfg_No; 
     
   -- 6.2 update lot cum number
	  UPDATE sys_param_conf_bt
         set ivalue2 = iLot_Cum -- + 1, add above
      WHERE param_id = 'LOT-01';                  SET stErrorLog = '092';
 
   -- 6.1 update tool status
    UPDATE oee_tool_bt set status = 'RUN'  WHERE tool_id = stTool_id;
*/


                                                  SET stErrorLog = '099';

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `order_trackin_box_sintb_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `order_trackin_box_sintb_sp`(stTool_Id VARCHAR(64),stMFG_NO VARCHAR(64),stLot_Id varchar(64),stArray varchar(255),
   iLot_Size int,iNum_Default int,iNum_Last int, iUnit_Type int)
    COMMENT 'track in box porcedure - 2020/05/05 lkchena'
label_sp: 
BEGIN
-- tip: order_trackin_box_sintb_sp for forming station, others use order_trackin_cart_sp2 -- 2020/03/02 lkchena

-- test case: 2020/03/01 lkchena
/* 
 call order_trackin_box_sintb_sp('ASCNY06#B','T-MFG_NO-003','A0100104.0','ACS00301',365,120,8,0);
 call order_trackin_box_sintb_sp('ASCNY06#B','T-MFG_NO-003','A0100104.0','ACS00301,ACS00302,ACS00303,ACS00305',365,120,8,0);
 select * from ker_wip_w0_bt order by report_time desc
 SELECT * FROM sys_job_log_bt ORDER BY time_start desc
*/
-- zero record -- 2020/03/04 lkchena
-- delete from ker_wip_bt 
-- -- select * from ker_wip_bt 
-- where tool_id = 'AF35Y01'
 

  DECLARE stJob_Name varchar(32) DEFAULT "order_trackin_box_sintb_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  
  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  DECLARE stBox_No  varchar(16) DEFAULT NULL;
  -- DECLARE iLot_Size2 int DEFAULT 0; -- we can't get existed lot size!!! final lot_size should different with lot_size original 2020/05/05 lkchena
  DECLARE iBox_Size  int DEFAULT 0;

  -- DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  -- DECLARE stMfg_No varchar(16) DEFAULT NULL;
  -- DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  -- lot parameter  2020/03/01 lkchena
  DECLARE stHeader  varchar(1) DEFAULT NULL;
  DECLARE stNode_Id varchar(2) DEFAULT NULL; -- 00 ~ zz
  DECLARE iLot_Len int DEFAULT 0;
  DECLARE iLot_Cum int DEFAULT 0;
  
  -- DECLARE stLot_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  DECLARE stCart_Id  varchar(16) DEFAULT NULL; -- 2020/03/02 lkchena 
  -- DECLARE iLot_Size  int DEFAULT 0;  
  DECLARE stRfid_Tag varchar(16) DEFAULT NULL;  

  -- dm use
  DECLARE iRD_Flag      int DEFAULT 0;
  DECLARE stOpe_No      varchar(16) DEFAULT NULL;
  DECLARE stPart_Id     varchar(24) DEFAULT NULL;
  DECLARE stMold_Id     varchar(24) DEFAULT NULL;
  DECLARE stPowder_Type varchar(16) DEFAULT NULL;
  DECLARE stPowder_Id   varchar(24) DEFAULT NULL;
  DECLARE stSTB_Plan    varchar(19) DEFAULT NULL;
  DECLARE stOrder_No    varchar(24) DEFAULT NULL; -- ? ORDER
  
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime; 
  
  DECLARE stRemain TEXT; -- array use
  DECLARE spilter CHAR(1);
  DECLARE pos9 INT DEFAULT 1 ;
  DECLARE str9 VARCHAR(1000);

  DECLARE doneCursor int DEFAULT 0;
                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';

/*
  -- 1. get lot parameter: 
   SELECT value1 as header,value2 as node_id,ivalue1 as lot_len,ivalue2 as lot_cum 
         into stHeader,stNode_Id,iLot_Len,iLot_Cum
   FROM sys_param_conf_bt where param_id = 'LOT-01'; SET stErrorLog = '001';
   
                               IF stHeader is NULL THEN -- no setting 
                                      SET stErrorLog = 'no sys_param_conf_bt data'; signal sqlstate '45000' set message_text = stErrorLog;
                                      leave label_sp;     
							   END IF;
   -- 1.2 get order base info 
    select  rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   date_stb,   order_no into
		   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No
      from tool_order_bt 
    where 1=1
     and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
     and mfg_no = stMfg_No;                           SET stErrorLog = '003';

   -- 1.8 ker variant
   SET dREPORT_TIME9 = NOW();  
   
   -- qqq: user is not full ready, so we use command part to replace it   
   -- 1.9 use command part
   set stPart_Id = 'ZZZ-01';
   set stMold_Id = 'ZZZ-01';
*/            
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;
  
WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '610';
	  -- 3.1.0
      set stBox_No  = str9;
      set iBox_Size = iNum_Default;
      if ( pos9 = 0 ) and ( iNum_Last <> 0 ) then set iBox_Size = iNum_Last; end if;
     
     select stBox_No, iLot_Size, iNum_Default, iNum_Last,iBox_Size;
     
/*      -- 3.1.1 add lot cum and get lot_id
      set iLot_Cum = iLot_Cum + 1;
      set stTmp = LPAD( iLot_Cum, iLot_Len, '0' );
      set stLot_Id = concat(stHeader,stNode_Id,stTmp); -- test:  select stLot_Id;
   
      -- 3.1.2 get cart_id, lot_size / '5060041'
      select a.cart_id, b.cnt_total into stCart_Id, iLot_Size   -- a.cart_type, , b.cnt_box
         from mcs_cart_base_bt a left join mcs_cart_type_bt b on a.cart_type = b.cart_type
       where rfid_tag = stRfid_Tag and active = 1;     SET stErrorLog = '010';

      -- 5.1 insert ker
			 INSERT INTO ker_wip_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, -- 
                                      cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id,claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                      part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                                      ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                                      )
                  SELECT dREPORT_TIME9, 'RT',stLot_Id,stLot_Id,'R' AS s, iLot_Size, iLot_Size, 350, -- 
                        stCart_Id,stTool_Id, stOrder_No, stMfg_No, stSTB_Plan, stPowder_Type, stPowder_Id, now(), now(),-- iLOT_SIZE_SPEC9,iCAST_SPEC9,iCAST_PCS_SPEC9,
                        part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                        ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                   FROM dm_flow_bt 
                  WHERE part_id = stPart_Id -- 'ZZZ-01' / '100.00'
                    AND  ope_no = stOpe_No;            SET stErrorLog = '011';
   
	  -- FOR DEBUG: SELECT  concat(stPart_Id,' - ',  stLot_Id,' - ', stRfid_Tag,' - ',stCart_Id,' - ',iLot_Size,' - ',stPowder_Type );
*/
                                                  SET stErrorLog = '620';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '691';

/*
   -- 6.1 update too_order_bt.status1 = 6 ???
   --     seems not to update is ok, if update to 6, then track-out need rollback to 5 ...
   --     ....
      update tool_order_bt 
        set status1 = 6
      where 1=1
       and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
       and mfg_no = stMfg_No; 
     
   -- 6.2 update lot cum number
	  UPDATE sys_param_conf_bt
         set ivalue2 = iLot_Cum -- + 1, add above
      WHERE param_id = 'LOT-01';                  SET stErrorLog = '092';
 
   -- 6.1 update tool status
    UPDATE oee_tool_bt set status = 'RUN'  WHERE tool_id = stTool_id;
*/


                                                  SET stErrorLog = '099';

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `order_trackin_box_stb_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `order_trackin_box_stb_sp`(stTool_Id VARCHAR(64),stMFG_NO VARCHAR(64),stLot_Id varchar(64),stArray varchar(255),
   iLot_Size int,iNum_Default int,iNum_Last int, iUnit_Type int, iWafer_Level int)
    COMMENT 'track in box porcedure - 2020/05/05 lkchena'
label_sp: 
BEGIN
-- test case: 2020/03/01 lkchena
/* 
 call order_trackin_box_stb_sp('AF35Y01','MFG_NO-107','A01000294.0','ACS00301',365,120,8,0,0);
 call order_trackin_box_stb_sp('AF35Y01','MFG_NO-107','A01000294.0','ACS00301,ACS00302,ACS00303,ACS00305',365,120,8,0,0);

 select * from ker_wip_w0_bt order by claim_time desc, lot_id desc
 select * from ker_wip_w2_bt order by claim_time desc, lot_id desc
 
 SELECT * FROM sys_job_log_bt ORDER BY time_start desc
*/
-- zero record -- 2020/03/04 lkchena
-- delete from ker_wip_bt 
-- -- select * from ker_wip_bt 
-- where tool_id = 'AF35Y01'
 

  DECLARE stJob_Name varchar(32) DEFAULT "order_trackin_box_stb_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  
  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  DECLARE stBox_Id        varchar(64) DEFAULT NULL;
  DECLARE stBox_Id_Start  varchar(64) DEFAULT NULL; -- for delete 2020/05/05 lkchena
  DECLARE stBox_No        varchar(64) DEFAULT NULL;
  DECLARE iBOX_CUM        int default 0;
  DECLARE iWafer_Size     int DEFAULT 0;

  declare stLOT_HEADER   varchar(1) DEFAULT null;-- default use one char to be header 

  declare stNODE_HEADER  varchar(2) DEFAULT null;-- two char for be flag 2020/04/22
  declare iLOT_LEN       int DEFAULT 5; 
  declare iLOT_CUM       int DEFAULT 0; 
  declare stLOT_LEAD     varchar(64) DEFAULT null;-- remove .XXX -- 2020/05/02 LKCHENA

  declare stFull_Id   varchar(3072) DEFAULT NULL;
  declare stFull_Time varchar(3072) DEFAULT NULL;
  declare stFull_2d   varchar(3072) DEFAULT NULL;
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime; 
  
  DECLARE stRemain TEXT; -- array use
  DECLARE spilter CHAR(1);
  DECLARE pos9 INT DEFAULT 1 ;
  DECLARE str9 VARCHAR(1000);

  DECLARE doneCursor int DEFAULT 0;
                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';

    -- 0.0 delete temp table dbfoge can't debug, when break at
     select @@global.sql_safe_updates into iTmp;
     if iTmp <> 0 then set sql_safe_updates = 0; end if; -- ??? why always show the error -- my god 2020/04/22 lkchena
    
	 IF     iWafer_Level = 2 THEN delete from ker_wip_w2_tt;     -- need wafer data               
	 ELSEIF iWafer_Level = 0 THEN delete from ker_wip_w0_tt; -- normal omi
     END IF;                                 set stErrorLog = '001';     
  
    -- 0.1 get parameter
	set stLOT_LEAD = substr(stLot_Id,1,length(stLot_Id)-2); set stErrorLog = '010'; -- select stLOT_LEAD;-- 2020/05/02 LKCHENAselect stLOT_LEAD;
    
    -- 0.2 get box_id suffix max number
	 IF     iWafer_Level = 2 THEN select max(lot_id) into stTmp from ker_wip_w2_bt where lot_id_p1 = stLot_Id; -- 'A01000294.0'  -- need wafer data 
	 ELSEIF iWafer_Level = 0 THEN select max(lot_id) into stTmp from ker_wip_w0_bt where lot_id_p1 = stLot_Id; -- 'A01000294.0'
     END IF;                                 set stErrorLog = '011';     
      
    -- 0.2.1 -- test: set stTmp = 'A01000294.09';
    SET iBOX_CUM = 0;
    if LENGTH(stTmp) > 0 THEN -- SELECT stTmp;

      -- 0.2.2 get , pos
      set iTmp     = INSTR(stTmp,'.'); -- SELECT stTmp, iTmp;
      set stTmp    = substr(stTmp,iTmp+1,2);    
      
-- qqq: here need decode function -- 2020/05/05 lkchena      
	  set iBOX_CUM = CONVERT(stTmp, UNSIGNED INTEGER)+1;
    
    END IF;    
    
    -- 0.2.3 for delete use
	set stBox_Id_Start = fn_cr8_box_id(stLOT_LEAD,iBOX_CUM,0);
	-- SELECT stBox_Id, iBOX_CUM;
          
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;
  
WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         
	  -- 3.1.0
      set stBox_No  = str9;                           SET stErrorLog = '610';
      set iWafer_Size = iNum_Default;
      if ( pos9 = 0 ) and ( iNum_Last <> 0 ) then set iWafer_Size = iNum_Last; end if;
      
      set stBox_Id = fn_cr8_box_id(stLOT_LEAD,iBOX_CUM,0);  SET stErrorLog = '620';
      set iBOX_CUM = iBOX_CUM + 1; -- for next run -- select stBox_No,stBox_Id, iLot_Size, iNum_Default, iNum_Last,iWafer_Size;
     
   	  IF     iWafer_Level = 2 THEN SET stErrorLog = '630';
       -- -------------------------------------------------------------------------
            set stFull_Id   = null;
            set stFull_Time = null;
            set stFull_2d   = null;

                   insert into ker_wip_w2_tt
                       (     report_time,cate,lot_id,lot_id_p1,lot_id_p2,lot_id_p3,
                             cart_no,box_no,lot_size,box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,box_label,s,pri,part_id,part_raw,route_id,raw_2d_code,
                             ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,claim_time,claim_user,
                             tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                             area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                             mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                             mold_id,powder_id,powder_type,cnt_plan,cnt_cur,cnt_act,cnt_ng,cnt_qc,cnt_test,cnt_tool,cnt_empty,
                             date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                             full_id,full_time,full_2d
						)
                  SELECT now() as report_time,cate,
                           stBox_Id as lot_id, stLot_Id as lot_id_p1, stBox_Id as lot_id_p2, lot_id_p3,-- 'A01000337.00'
                                 cart_no,
                           stBox_No as box_no,
                           iWafer_Size as lot_size,
                                 box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,
                           null as box_label, -- stBox_Label
                                 s,pri,part_id,part_raw,route_id,raw_2d_code,
                                 ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,now(),claim_user, -- claim_time,
                                 tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                                 area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                                 mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                                 mold_id,powder_id,powder_type,
                           -- //keep wip information -- 2020/05/02 lkchena-- cnt_plan,cnt_cur,cnt_act, cnt_ng,cnt_qc,cnt_test,
						   -- use current wafer_size 2020/05/05 lkchena
                           iWafer_Size as cnt_plan, iWafer_Size as cnt_cur, iWafer_Size as cnt_act,
                                 cnt_ng,cnt_qc,cnt_test,cnt_tool, cnt_empty,
                                 date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                          stFull_Id,stFull_Time,stFull_2d
                   FROM ker_wip_bt
                  where lot_id = stLot_Id
                  limit 1;                                 set stErrorLog = '639';
       -- -------------------------------------------------------------------------
	  ELSEIF iWafer_Level = 0 THEN SET stErrorLog = '650';
       -- -------------------------------------------------------------------------
            set stFull_Id   = null;
            set stFull_Time = null;
            set stFull_2d   = null;

                   insert into ker_wip_w0_tt
                       (     report_time,cate,lot_id,lot_id_p1,lot_id_p2,lot_id_p3,
                             cart_no,box_no,lot_size,box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,box_label,s,pri,part_id,part_raw,route_id,raw_2d_code,
                             ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,claim_time,claim_user,
                             tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                             area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                             mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                             mold_id,powder_id,powder_type,cnt_plan,cnt_cur,cnt_act,cnt_ng,cnt_qc,cnt_test,cnt_tool,cnt_empty,
                             date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                             full_id,full_time,full_2d
						)
                  SELECT now() as report_time,cate,
                           stBox_Id as lot_id, stLot_Id as lot_id_p1, stBox_Id as lot_id_p2, lot_id_p3,-- 'A01000337.00'
                                 cart_no,
                           stBox_No as box_no,
                           iWafer_Size as lot_size,
                                 box_cnt,lot_size_spec,box_size_spec,wafer_size_spec,
                           null as box_label, -- stBox_Label
                                 s,pri,part_id,part_raw,route_id,raw_2d_code,
                                 ope_no,ope_name,stage_id,stage_name,stage_order,ope_cate,extra_step,now(),claim_user, -- claim_time,
                                 tool_grp_id,tool_grp,ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func,er,
                                 area_id,area_name,pos_id,pos_desc,tool_id,pre_main_tool,tag_track_io,proc_time,cust_id,cust_name,order_no,
                                 mfg_no,rd_flag,b_vendor_id,b_vendor_name,lot_note,lot_memo,track_in_time,track_out_time,proc_start_time,proc_end_time,
                                 mold_id,powder_id,powder_type,
                           -- //keep wip information -- 2020/05/02 lkchena-- cnt_plan,cnt_cur,cnt_act, cnt_ng,cnt_qc,cnt_test,
						   -- use current wafer_size 2020/05/05 lkchena
                           iWafer_Size as cnt_plan, iWafer_Size as cnt_cur, iWafer_Size as cnt_act,
                                 cnt_ng,cnt_qc,cnt_test,cnt_tool, cnt_empty,
                                 date_stb,date_stb_real,date_due,date_late,date_comp,split_from,merge_from,
                          stFull_Id,stFull_Time,stFull_2d
				  FROM ker_wip_bt
                  where lot_id = stLot_Id
                  limit 1;                                 set stErrorLog = '619';
       -- -------------------------------------------------------------------------
      END IF;                                     SET stErrorLog = '620';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '691';


    -- 9. write to final table
		IF     iWafer_Level = 2 THEN      set stErrorLog = '710'; -- need wafer data             

            delete from ker_wip_w2_bt where lot_id_p1 = stLot_Id and lot_id >= stBox_Id_Start;
            insert into ker_wip_w2_bt select * from ker_wip_w2_tt;
            
		ELSEIF iWafer_Level = 0 THEN      set stErrorLog = '720';-- normal omi
            
            delete from ker_wip_w0_bt where lot_id_p1 = stLot_Id and lot_id >= stBox_Id_Start;
            insert into ker_wip_w0_bt select * from ker_wip_w0_tt;
         
        END IF;                           set stErrorLog = '799'; -- ELSE


                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `order_trackin_cart_sp1` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `order_trackin_cart_sp1`(IN stTool_Id varchar(12), IN stMfg_No varchar(24), IN stArray varchar(255))
    COMMENT 'track in cart porcedure - 2020/03/01 lkchena'
label_sp: 
BEGIN
-- tip: order_trackin_cart_sp1 for forming station, others use order_trackin_cart_sp2 -- 2020/03/02 lkchena

-- test case: 2020/03/01 lkchena
-- call order_trackin_cart_sp1('AF35Y01','T-MFG_NO-003','5060041');
-- call order_trackin_cart_sp1('AF35Y01','T-MFG_NO-003','5060041,1000022');
-- select * from ker_wip_bt order by report_time desc
-- SELECT * FROM sys_job_log_bt ORDER BY time_start desc

-- zero record -- 2020/03/04 lkchena
-- delete from ker_wip_bt 
-- -- select * from ker_wip_bt 
-- where tool_id = 'AF35Y01'
 

  DECLARE stJob_Name varchar(32) DEFAULT "order_trackin_cart_sp1";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  -- DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  -- DECLARE stMfg_No varchar(16) DEFAULT NULL;
  -- DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  -- lot parameter  2020/03/01 lkchena
  DECLARE stHeader  varchar(1) DEFAULT NULL;
  DECLARE stNode_Id varchar(2) DEFAULT NULL; -- 00 ~ zz
  DECLARE iLot_Len int DEFAULT 0;
  DECLARE iLot_Cum int DEFAULT 0;
  
  DECLARE stLot_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  DECLARE stCart_Id  varchar(16) DEFAULT NULL; -- 2020/03/02 lkchena 
  DECLARE iLot_Size  int DEFAULT 0;  
  DECLARE stRfid_Tag varchar(16) DEFAULT NULL;  

  -- dm use
  DECLARE iRD_Flag      int DEFAULT 0;
  DECLARE stOpe_No      varchar(16) DEFAULT NULL;
  DECLARE stPart_Id     varchar(24) DEFAULT NULL;
  DECLARE stMold_Id     varchar(24) DEFAULT NULL;
  DECLARE stPowder_Type varchar(16) DEFAULT NULL;
  DECLARE stPowder_Id   varchar(24) DEFAULT NULL;
  DECLARE stSTB_Plan    varchar(19) DEFAULT NULL;
  DECLARE stOrder_No    varchar(24) DEFAULT NULL; -- ? ORDER
  
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime; 
  
  DECLARE stRemain TEXT; -- array use
  DECLARE spilter CHAR(1);
  DECLARE pos9 INT DEFAULT 1 ;
  DECLARE str9 VARCHAR(1000);

  DECLARE doneCursor int DEFAULT 0;
/*
  DECLARE cur1 CURSOR FOR
  SELECT a.mfg_no, a.mold_id, a.tool_id
  FROM erp_prod_stb_t9_prod a
    INNER JOIN tool_order_bt b
      ON a.mfg_no = b.mfg_no
      AND a.mold_id = b.mold_id
      AND b.status1 <> 0
      and b.rd_flag = 0 and a.rd_flag = 0; 
*/
                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';

  -- 1. get lot parameter: 
   SELECT value1 as header,value2 as node_id,ivalue1 as lot_len,ivalue2 as lot_cum 
         into stHeader,stNode_Id,iLot_Len,iLot_Cum
   FROM sys_param_conf_bt where param_id = 'LOT-01'; SET stErrorLog = '001';
   
                               IF stHeader is NULL THEN -- no setting 
                                      SET stErrorLog = 'no sys_param_conf_bt data'; signal sqlstate '45000' set message_text = stErrorLog;
                                      leave label_sp;     
							   END IF;
   -- 1.2 get order base info 
    select  rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   date_stb,   order_no into
		   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No
      from tool_order_bt 
    where 1=1
     and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
     and mfg_no = stMfg_No;                           SET stErrorLog = '003';

   -- 1.8 ker variant
   SET dREPORT_TIME9 = NOW();  
   
   -- qqq: user is not full ready, so we use command part to replace it   
   -- 1.9 use command part
   set stPart_Id = 'ZZZ-01';
   set stMold_Id = 'ZZZ-01';
            
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;

WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '010';
	  -- 3.1.0
      set stRfid_Tag = str9;
   
      -- 3.1.1 add lot cum and get lot_id
      set iLot_Cum = iLot_Cum + 1;
      set stTmp = LPAD( iLot_Cum, iLot_Len, '0' );
      set stLot_Id = concat(stHeader,stNode_Id,stTmp); -- test:  select stLot_Id;
   
      -- 3.1.2 get cart_id, lot_size / '5060041'
      select a.cart_id, b.cnt_total into stCart_Id, iLot_Size   -- a.cart_type, , b.cnt_box
         from mcs_cart_base_bt a left join mcs_cart_type_bt b on a.cart_type = b.cart_type
       where rfid_tag = stRfid_Tag and active = 1;     SET stErrorLog = '010';

      -- 5.1 insert ker
			 INSERT INTO ker_wip_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, -- 
                                      cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id,claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                      part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                                      ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                                      )
                  SELECT dREPORT_TIME9, 'RT',stLot_Id,stLot_Id,'R' AS s, iLot_Size, iLot_Size, 350, -- 
                        stCart_Id,stTool_Id, stOrder_No, stMfg_No, stSTB_Plan, stPowder_Type, stPowder_Id, now(), now(),-- iLOT_SIZE_SPEC9,iCAST_SPEC9,iCAST_PCS_SPEC9,
                        part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                        ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func -- 2020/03/19 lkchena
                   FROM dm_flow_bt 
                  WHERE part_id = stPart_Id -- 'ZZZ-01' / '100.00'
                    AND  ope_no = stOpe_No;            SET stErrorLog = '011';
   
	  -- FOR DEBUG: SELECT  concat(stPart_Id,' - ',  stLot_Id,' - ', stRfid_Tag,' - ',stCart_Id,' - ',iLot_Size,' - ',stPowder_Type );

                                                  SET stErrorLog = '020';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '091';

   -- 6.1 update too_order_bt.status1 = 6 ???
   --     seems not to update is ok, if update to 6, then track-out need rollback to 5 ...
   --     ....
      update tool_order_bt 
        set status1 = 6
      where 1=1
       and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
       and mfg_no = stMfg_No; 
     
   -- 6.2 update lot cum number
	  UPDATE sys_param_conf_bt
         set ivalue2 = iLot_Cum -- + 1, add above
      WHERE param_id = 'LOT-01';                  SET stErrorLog = '092';
 
   -- 6.1 update tool status
    UPDATE oee_tool_bt set status = 'RUN'  WHERE tool_id = stTool_id;



                                                  SET stErrorLog = '099';

/*  
  OPEN cur1;
  REPEAT
    FETCH cur1 INTO stMfg_No, stMold_Id, stTool_Id;
    IF NOT doneCursor THEN
      SET stErrorLog = CONCAT('030_', stMfg_No);
      
      DELETE FROM erp_prod_stb_t9_prod
      WHERE mfg_no = stMfg_No
        AND mold_id = stMold_Id 
        and rd_flag = 0;
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CLOSE cur1;
  SET stErrorLog = CONCAT('061_', 'insert compact table');

  
  DELETE FROM tool_order_bt
  WHERE (mfg_no, mold_id) IN (SELECT mfg_no,mold_id FROM erp_prod_stb_t9_prod)
    and rd_flag = 0; 

  SET stErrorLog = '070';

  
  INSERT INTO tool_order_bt (pri, status1, tool_grp_id, tool_id, part_id, part_raw, part_name, part_erp_no, mold_id, powder_type, powder_erp_no, cnt_plan, cnt_act, date_due, date_stb, date_comp, mfg_no, order_no, rd_flag, cust_id, weight, unit, owner_id, owner_name, note, rec_user, rec_time,
                             time_order, ope_no, ope_name, stage_id, stage_name, stage_order) 
    SELECT pri,status1,tool_grp_id,tool_id,part_id,part_raw,part_name,part_erp_no,mold_id,powder_type,powder_erp_no,cnt_plan,cnt_act,date_due,date_stb,date_comp,mfg_no,order_no,rd_flag,cust_id,weight,unit,owner_id,owner_name,note,rec_user,rec_time,
           DATE_FORMAT(NOW(),"%Y/%m/%d %H:00:00") AS time_order,'100.00','成型作業','Forming','成型站', '100' 
          
    FROM erp_prod_stb_t9_prod
  where 1=1
    and rd_flag = 0; 
  
  
*/  

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `svid_takeover_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `svid_takeover_sp`(IN stTool_Id varchar(12), IN stPart_Id varchar(24), IN stMold_Id varchar(16), IN stTime_Order varchar(19))
BEGIN

  
  
  

  DECLARE stJob_Name varchar(32) DEFAULT "svid_takeover_sp";
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  
  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  DECLARE stPart_Id2 varchar(24) DEFAULT NULL; 

  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    SELECT
      0 INTO iResult_1_ok_0_ng;
    ROLLBACK;
    CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
    SELECT
      iResult_1_ok_0_ng;
  END;                                        
  CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
  SET stErrorLog = '000';
  
  
  

  
  IF LENGTH(stTime_Order) = 0 THEN
    SET stTime_Order = DATE_FORMAT(NOW(), "%Y/%m/%d %H:%i:%S");
  END IF;

  
  set stPart_Id2 = stPart_Id;
  if LENGTH(stPart_Id) = 0 THEN
     set stPart_Id2 = "%";
  End IF;


  
  DELETE
    FROM svid_data_bth 
  WHERE 1 = 1
    AND tool_id = stTool_Id 
    AND part_id like stPart_Id2 
    AND mold_id = stMold_Id 
    AND kind = 2
    AND time_order = stTime_Order;

  INSERT INTO svid_data_bth
    SELECT
      kind,
      stTime_Order, 
      ver,
      ver_act,
      mfg_no,
      rd_flag,
      tool_id,
      part_id,
      mold_id,
      recipe_id,
      ope_no,
      active,
      igroup,
      iord,
      sour1,
      nx1,
      field_name1,
      field_desc1,
      value1,
      value1s,
      unit1,
      csl1,
      sign1,
      lsl1,
      usl1,
      sour2,
      nx2,
      field_name2,
      field_desc2,
      value2,
      value2s,
      unit2,
      csl2,
      sign2,
      lsl2,
      usl2,
      sour3,
      nx3,
      field_name3,
      field_desc3,
      value3,
      value3s,
      unit3,
      csl3,
      sign3,
      lsl3,
      usl3,
      sour1a,
      nx1a,
      field_name1a,
      field_desc1a,
      value1a,
      value1as,
      sour1b,
      nx1b,
      field_name1b,
      field_desc1b,
      value1b,
      value1bs,
      freq,
      note,
      rec_user,
      rec_time,
      itag
    FROM svid_data_bt
    WHERE 1 = 1
    AND tool_id = stTool_Id 
    AND part_id like stPart_Id2 
    AND mold_id = stMold_Id 
    AND kind = 2
    ORDER BY igroup ASC, iord ASC;
  SET stErrorLog = CONCAT('010_', '00');

  DELETE
    FROM svid_data_bt
  WHERE 1 = 1
    AND tool_id = stTool_Id 
    AND part_id like stPart_Id2 
    AND mold_id = stMold_Id 
    AND kind = 2;
  SET stErrorLog = CONCAT('020_', '00');

  
  INSERT INTO svid_data_bt
    SELECT
      2 AS kind,
      stTime_Order, 
      ver,
      ver_act,
      mfg_no,
      rd_flag,
      tool_id,
      part_id,
      mold_id,
      recipe_id,
      ope_no,
      active,
      igroup,
      iord,
      sour1,
      nx1,
      field_name1,
      field_desc1,
      value1,
      value1s,
      unit1,
      csl1,
      sign1,
      lsl1,
      usl1,
      sour2,
      nx2,
      field_name2,
      field_desc2,
      value2,
      value2s,
      unit2,
      csl2,
      sign2,
      lsl2,
      usl2,
      sour3,
      nx3,
      field_name3,
      field_desc3,
      value3,
      value3s,
      unit3,
      csl3,
      sign3,
      lsl3,
      usl3,
      sour1a,
      nx1a,
      field_name1a,
      field_desc1a,
      value1a,
      value1as,
      sour1b,
      nx1b,
      field_name1b,
      field_desc1b,
      value1b,
      value1bs,
      freq,
      note,
      rec_user,
      rec_time,
      itag
    FROM svid_data_bt
    WHERE 1 = 1
    AND tool_id = stTool_Id 
    AND part_id like stPart_Id2 
    AND mold_id = stMold_Id 
    AND kind = 3
    ORDER BY igroup ASC, iord ASC;
  SET stErrorLog = CONCAT('030_', '00');

  DELETE
    FROM svid_data_bt
  WHERE 1 = 1
    AND tool_id = stTool_Id 
    AND part_id like stPart_Id2 
    AND mold_id = stMold_Id 
    AND kind = 3;
  SET stErrorLog = CONCAT('042_', '00');

  
  
  SELECT
    COUNT(*) INTO iCNT
  FROM svid_data_bt
  WHERE 1 = 1
  AND tool_id = stTool_Id 
  AND part_id like stPart_Id2 
  AND mold_id = stMold_Id 
  AND kind = 2;

  
  IF iCNT = 0 THEN

    INSERT INTO svid_data_bt
      SELECT
        3 AS kind,
        DATE_FORMAT(NOW(), "%Y/%m/%d %H:%i:%S") AS time_order,
        ver,
        ver_act,
        mfg_no,
        rd_flag,
        tool_id,
        part_id,
        mold_id,
        recipe_id,
        ope_no,
        active,
        igroup,
        iord,
        sour1,
        nx1,
        field_name1,
        field_desc1,
        value1,
        value1s,
        unit1,
        csl1,
        sign1,
        lsl1,
        usl1,
        sour2,
        nx2,
        field_name2,
        field_desc2,
        value2,
        value2s,
        unit2,
        csl2,
        sign2,
        lsl2,
        usl2,
        sour3,
        nx3,
        field_name3,
        field_desc3,
        value3,
        value3s,
        unit3,
        csl3,
        sign3,
        lsl3,
        usl3,
        sour1a,
        nx1a,
        field_name1a,
        field_desc1a,
        value1a,
        value1as,
        sour1b,
        nx1b,
        field_name1b,
        field_desc1b,
        value1b,
        value1bs,
        freq,
        note,
        rec_user,
        rec_time,
        itag
      FROM svid_data_bt
      WHERE 1 = 1
      AND tool_id = stTool_Id 
      AND part_id like stPart_Id2 
      AND mold_id = stMold_Id 
      AND kind = 1
      AND ver_act = 1
      ORDER BY igroup ASC, iord ASC;

  
  ELSEIF iCNT <> 0 THEN

    INSERT INTO svid_data_bt
      SELECT
        3 AS kind,
        DATE_FORMAT(NOW(), "%Y/%m/%d %H:%i:%S") AS time_order,
        ver,
        ver_act,
        mfg_no,
        rd_flag,
        tool_id,
        part_id,
        mold_id,
        recipe_id,
        ope_no,
        active,
        igroup,
        iord,
        sour1,
        nx1,
        field_name1,
        field_desc1,
        value1,
        value1s,
        unit1,
        csl1,
        sign1,
        lsl1,
        usl1,
        sour2,
        nx2,
        field_name2,
        field_desc2,
        value2,
        value2s,
        unit2,
        csl2,
        sign2,
        lsl2,
        usl2,
        sour3,
        nx3,
        field_name3,
        field_desc3,
        value3,
        value3s,
        unit3,
        csl3,
        sign3,
        lsl3,
        usl3,
        sour1a,
        nx1a,
        field_name1a,
        field_desc1a,
        value1a,
        value1as,
        sour1b,
        nx1b,
        field_name1b,
        field_desc1b,
        value1b,
        value1bs,
        freq,
        note,
        rec_user,
        rec_time,
        itag
      FROM svid_data_bt
      WHERE 1 = 1
      AND tool_id = stTool_Id 
      AND part_id like stPart_Id2 
      AND mold_id = stMold_Id 
      AND kind = 2
      AND ver_act = 1
      ORDER BY igroup ASC, iord ASC;

  END IF;


  SET stErrorLog = '900'; 
  SELECT
    1 INTO iResult_1_ok_0_ng; 
  SELECT
    iResult_1_ok_0_ng; 
  CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `sys_job_exec_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `sys_job_exec_sp`(
	IN `iJob_Satsu` INT,
	IN `stJob_Name` VARCHAR(32),
	IN `dTime_Start` DATETIME,
	IN `iElp_Spec` INT,
	IN `stNote` VARCHAR(128) 
)
BEGIN
-- 2018/04/11 lkchena
-- iJob_Status: 1: under execute 9:compelete job -3: over spec -9:error

-- test case:
/*
  CALL ker_wip_snapshot_sp('YES',get_shift(NOW(),1) );
  SELECT SLEEP(2);
  CALL ker_wip_snapshot_sp('YES',get_shift(NOW(),2) );
  SELECT SLEEP(2);
 CALL ker_wip_snapshot_sp('YES',get_shift(NOW(),3) );
  SELECT SLEEP(2);

  -- SELECT * FROM sys_job_log_bt a ORDER BY a.time_start desc
*/

START TRANSACTION;
 
  IF iJob_Satsu = 1 THEN

    INSERT INTO sys_job_log_bt( job_name,time_start,elp_spec,job_status,note ) VALUES
                              (stJob_Name, dTime_Start,iElp_Spec,iJob_Satsu,stNote );

  ELSEIF iJob_Satsu = 9 THEN -- change to use 9: 2020/03/01 lkchena
    
     UPDATE sys_job_log_bt 
        SET timer_end = SYSDATE(), 
            elp = TIME_TO_SEC(TIMEDIFF(SYSDATE(), dTime_Start)), 
            job_status = iJob_Satsu,
            note = stNote  -- 2020/04/22 lkchena
        WHERE job_name = stJob_Name
          AND time_start = dTime_Start;
      
  ELSEIF iJob_Satsu = -9 THEN

     UPDATE sys_job_log_bt 
        SET timer_error = SYSDATE(),
            elp = TIME_TO_SEC(TIMEDIFF(SYSDATE(), dTime_Start)), 
            job_status = iJob_Satsu,
            note = stNote       
        WHERE job_name = stJob_Name
          AND time_start = dTime_Start;

  end IF;

 
 Commit;
 
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `tcs_tool_er_chg_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `tcs_tool_er_chg_sp`(
	IN `stTool_id` VARCHAR(12),
	IN `stDevice` VARCHAR(16),
	IN `stER` VARCHAR(16),
	IN `stUser_id` VARCHAR(16),
	IN `stMemo` VARCHAR(64)
)
BEGIN
DECLARE cr_stack_depth_handler INTEGER ;
DECLARE cr_stack_depth INTEGER DEFAULT cr_debug.ENTER_MODULE2('tcs_tool_er_chg_sp', 'mes_dev', 7, 100633) ;















  declare  stJob_Name varchar(32) DEFAULT 'tcs_tool_er_chg_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stErrorLog varchar(64);
   
  declare stPreER varchar(16);

  declare stLP_CH_1 VARCHAR(16); 
  declare stLP_CH_2 VARCHAR(16); 
  declare stLP_CH_3 VARCHAR(16); 
  declare stLP_CH_4 VARCHAR(16);

                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
SET cr_stack_depth_handler = cr_stack_depth ;
SET cr_stack_depth = cr_debug.ENTER_HANDLER('tcs_tool_er_chg_sp_Handler', 'tcs_tool_er_chg_sp', 'mes_dev', 7, 100633) ;
                                                      CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'INT', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stPreER', stPreER, 'varchar(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_1', stLP_CH_1, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_2', stLP_CH_2, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_3', stLP_CH_3, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_4', stLP_CH_4, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stTool_id`', `stTool_id`, 'VARCHAR(12)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stDevice`', `stDevice`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stER`', `stER`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stUser_id`', `stUser_id`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stMemo`', `stMemo`, 'VARCHAR(64)', cr_stack_depth) ;
CALL cr_debug.TRACE(38, 38, 50, 55, cr_stack_depth) ;
CALL cr_debug.TRACE(39, 39, 54, 86, cr_stack_depth) ;
SELECT 0 into iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, '', cr_stack_depth) ; 
                                                      CALL cr_debug.TRACE(40, 40, 54, 63, cr_stack_depth) ;
ROLLBACK;
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'INT', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stPreER', stPreER, 'varchar(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_1', stLP_CH_1, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_2', stLP_CH_2, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_3', stLP_CH_3, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_4', stLP_CH_4, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stTool_id`', `stTool_id`, 'VARCHAR(12)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stDevice`', `stDevice`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stER`', `stER`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stUser_id`', `stUser_id`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stMemo`', `stMemo`, 'VARCHAR(64)', cr_stack_depth) ;     
                                                              CALL cr_debug.TRACE(41, 41, 62, 130, cr_stack_depth) ;
call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog);
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ; 
                                                      CALL cr_debug.TRACE(42, 42, 54, 79, cr_stack_depth) ;
SELECT iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ; 
                                                  CALL cr_debug.TRACE(43, 43, 50, 53, cr_stack_depth) ;
CALL cr_debug.LEAVE_MODULE(cr_stack_depth - 1) ;
end;                                        
                                                              CALL cr_debug.UPDATE_WATCH3('`stTool_id`', `stTool_id`, 'VARCHAR(12)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stDevice`', `stDevice`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stER`', `stER`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stUser_id`', `stUser_id`, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('`stMemo`', `stMemo`, 'VARCHAR(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, 'varchar(32)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, 'int', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, 'datetime', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, 'INT', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, 'varchar(64)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stPreER', stPreER, 'varchar(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_1', stLP_CH_1, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_2', stLP_CH_2, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_3', stLP_CH_3, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('stLP_CH_4', stLP_CH_4, 'VARCHAR(16)', cr_stack_depth) ;
CALL cr_debug.TRACE(8, 8, 0, 5, cr_stack_depth) ;
CALL cr_debug.TRACE(44, 44, 62, 123, cr_stack_depth) ;
call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, '', cr_stack_depth) ;
                                                              CALL cr_debug.TRACE(45, 45, 62, 91, cr_stack_depth) ;
select '000' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;    
 
 CALL cr_debug.TRACE(47, 119, 1, 8, cr_stack_depth) ;
IF  STRCMP(stDevice,'MAIN') = 0 THEN 

                                                              CALL cr_debug.TRACE(49, 49, 62, 94, cr_stack_depth) ;
select '020-a1' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
    CALL cr_debug.TRACE(50, 50, 4, 22, cr_stack_depth) ;
START TRANSACTION;
         CALL cr_debug.TRACE(51, 51, 9, 70, cr_stack_depth) ;
UPDATE oee_tool_bt set  er = stER  WHERE tool_id = stTool_id;
CALL cr_debug.UPDATE_SYSTEM_CALLS(104) ;
    CALL cr_debug.TRACE(52, 52, 4, 11, cr_stack_depth) ;
COMMIT;
                                                              CALL cr_debug.TRACE(53, 53, 62, 94, cr_stack_depth) ;
select '020-a2' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
   
   
   
   
                                                              CALL cr_debug.TRACE(58, 58, 62, 94, cr_stack_depth) ;
select '020-a9' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;

 ELSEIF stDevice LIKE 'LP%' THEN
                                                              CALL cr_debug.TRACE(61, 61, 62, 94, cr_stack_depth) ;
select '020-b0' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;    
                                                              CALL cr_debug.TRACE(87, 87, 64, 96, cr_stack_depth) ;
select '020-b9' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;

 ELSEIF stDevice LIKE 'CH%' THEN

                                                              CALL cr_debug.TRACE(91, 91, 62, 94, cr_stack_depth) ;
select '020-c0' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;    
                                                              CALL cr_debug.TRACE(116, 116, 64, 96, cr_stack_depth) ;
select '020-c9' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;


 END IF;

 
                                                              CALL cr_debug.TRACE(122, 122, 62, 91, cr_stack_depth) ;
select '900' INTO stErrorLog;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('stErrorLog', stErrorLog, '', cr_stack_depth) ;
                                               
                                                         CALL cr_debug.TRACE(124, 124, 57, 89, cr_stack_depth) ;
SELECT 1 into iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ;
CALL cr_debug.UPDATE_WATCH3('iResult_1_ok_0_ng', iResult_1_ok_0_ng, '', cr_stack_depth) ; 
                                                         CALL cr_debug.TRACE(125, 125, 57, 82, cr_stack_depth) ;
SELECT iResult_1_ok_0_ng;
CALL cr_debug.UPDATE_SYSTEM_CALLS(101) ; 
                                                               CALL cr_debug.TRACE(126, 126, 63, 123, cr_stack_depth) ;
call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');
CALL cr_debug.UPDATE_WATCH3('stJob_Name', stJob_Name, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('dExec_Time', dExec_Time, '', cr_stack_depth) ;
CALL cr_debug.UPDATE_WATCH3('iElp_Spec', iElp_Spec, '', cr_stack_depth) ;


CALL cr_debug.TRACE(129, 129, 0, 3, cr_stack_depth) ;
CALL cr_debug.LEAVE_MODULE(cr_stack_depth - 1) ;
END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `tcs_tool_status_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `tcs_tool_status_sp`(
  IN stTool_id VARCHAR(12),	
  IN stDevice VARCHAR(16),
	IN stNewStatus VARCHAR(16),
	IN stUser_id VARCHAR(16),
	IN stMemo VARCHAR(64))
BEGIN
























  declare  stJob_Name varchar(32) DEFAULT "tcs_tool_status_sp";
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stErrorLog varchar(64);
   
  declare stPreStatus varchar(16);
  declare stStatus varchar(16); 

  declare stLP_CH_1 VARCHAR(16); 
  declare stLP_CH_2 VARCHAR(16); 
  declare stLP_CH_3 VARCHAR(16); 
  declare stLP_CH_4 VARCHAR(16);

                                                  declare exit handler for SQLEXCEPTION
                                                  BEGIN
                                                      SELECT 0 into iResult_1_ok_0_ng; 
                                                      ROLLBACK;     
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
                                                      SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              select '000' INTO stErrorLog;    

 
 IF  STRCMP(stDevice,'MAIN') = 0 THEN 

    
                                                              select '020-a0' INTO stErrorLog;    
    SELECT a.status INTO stPreStatus FROM oee_tool_bt a WHERE a.tool_id = stTool_id;
                                                              select '020-a1' INTO stErrorLog;
    set stStatus = stNewStatus; 
  	if length(stStatus) = 0 then set stStatus = stPreStatus; END IF;
		
	START TRANSACTION;
         UPDATE oee_tool_bt set status = stStatus  WHERE tool_id = stTool_id;
    COMMIT;
                                                              select '020-a2' INTO stErrorLog;
    START TRANSACTION;
         INSERT INTO oee_tool_sch_bt( tool_id,device,status,pre_status,note,claim_time,claim_user )
         VALUES (stTool_id,'MAIN',stStatus,stPreStatus,stMemo,NOW(),stUser_id);
    COMMIT;
                                                              select '020-a9' INTO stErrorLog;

 ELSEIF stDevice LIKE 'LP%' THEN
                                                              select '020-b0' INTO stErrorLog;    
    
    SELECT a.lp_1,a.lp_2,a.lp_3,a.lp_4 INTO stLP_CH_1,stLP_CH_2,stLP_CH_3,stLP_CH_4 
      FROM oee_tool_bt a WHERE a.tool_id = stTool_id;
                                                              select '020-b1' INTO stErrorLog; 
  
    IF     STRCMP(stDevice,'LP_1') = 0 THEN SELECT stLP_CH_1 INTO stPreStatus;
    ELSEIF STRCMP(stDevice,'LP_2') = 0 THEN SELECT stLP_CH_2 INTO stPreStatus; 
    ELSEIF STRCMP(stDevice,'LP_3') = 0 THEN SELECT stLP_CH_3 INTO stPreStatus; 
    ELSEIF STRCMP(stDevice,'LP_4') = 0 THEN SELECT stLP_CH_4 INTO stPreStatus; 
    END IF;
                                                              select '020-b2' INTO stErrorLog;                                                               
    
    IF     STRCMP(stDevice,'LP_1') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set lp_1 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    ELSEIF STRCMP(stDevice,'LP_2') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set lp_2 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    ELSEIF STRCMP(stDevice,'LP_3') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set lp_3 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    ELSEIF STRCMP(stDevice,'LP_4') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set lp_4 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    END IF;
                                                              select '020-b3' INTO stErrorLog;
    
    START TRANSACTION;
         INSERT INTO oee_tool_sch_bt( tool_id,device,status,pre_status,note,claim_time,claim_user )
         VALUES (stTool_id,stDevice,stStatus,stPreStatus,stMemo,NOW(),stUser_id);
    COMMIT;    
                                                              select '020-b9' INTO stErrorLog;

 ELSEIF stDevice LIKE 'CH%' THEN

                                                              select '020-c0' INTO stErrorLog;    
    
    SELECT a.ch_1,a.ch_2,a.ch_3,a.ch_4 INTO stLP_CH_1,stLP_CH_2,stLP_CH_3,stLP_CH_4 
      FROM oee_tool_bt a WHERE a.tool_id = stTool_id;
                                                              select '020-c1' INTO stErrorLog; 
  
    IF     STRCMP(stDevice,'CH_1') = 0 THEN SELECT stLP_CH_1 INTO stPreStatus;
    ELSEIF STRCMP(stDevice,'CH_2') = 0 THEN SELECT stLP_CH_2 INTO stPreStatus; 
    ELSEIF STRCMP(stDevice,'CH_3') = 0 THEN SELECT stLP_CH_3 INTO stPreStatus; 
    ELSEIF STRCMP(stDevice,'CH_4') = 0 THEN SELECT stLP_CH_4 INTO stPreStatus; 
    END IF;
                                                              select '020-c2' INTO stErrorLog;                                                               
    
    IF     STRCMP(stDevice,'CH_1') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set ch_1 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    ELSEIF STRCMP(stDevice,'CH_2') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set ch_2 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    ELSEIF STRCMP(stDevice,'CH_3') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set ch_3 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    ELSEIF STRCMP(stDevice,'CH_4') = 0 THEN START TRANSACTION;  UPDATE oee_tool_bt set ch_4 = stStatus  WHERE tool_id = stTool_id; COMMIT;
    END IF;
                                                              select '020-c3' INTO stErrorLog;
    
    START TRANSACTION;
         INSERT INTO oee_tool_sch_bt( tool_id,device,status,pre_status,note,claim_time,claim_user )
         VALUES (stTool_id,stDevice,stStatus,stPreStatus,stMemo,NOW(),stUser_id);
    COMMIT;    
                                                              select '020-c9' INTO stErrorLog;


 END IF;

 
                                                              select '900' INTO stErrorLog;
  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 9,stJob_Name,dExec_Time,iElp_Spec,'');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `tcs_tool_trackinout_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`%` PROCEDURE `tcs_tool_trackinout_sp`(
   IN stReader_id VARCHAR(6), 
   IN iIn_Out integer, 
   IN stRfid_Tag VARCHAR(16)
  )
BEGIN


















  declare  stJob_Name varchar(32) DEFAULT 'tcs_tool_trackinout_sp';
  declare  iElp_Spec int DEFAULT 30;
  declare  dExec_Time datetime DEFAULT NOW();  
  declare  iResult_1_ok_0_ng INT;
  declare  stErrorLog varchar(64);
   
  declare stTool_id varchar(16);
  declare stDevice  varchar(8);
  declare stLP      VARCHAR(16); 
  DECLARE stCart_id varchar(6);  
                                                  declare exit handler for SQLEXCEPTION
                                                  begin       SELECT 0 into iResult_1_ok_0_ng; ROLLBACK;                                                           
                                                              call sys_job_exec_sp(-9,stJob_Name,dExec_Time,iElp_Spec,stErrorLog); 
                                                              SELECT iResult_1_ok_0_ng; 
                                                  end;                                        
                                                              call sys_job_exec_sp( 1,stJob_Name, dExec_Time,iElp_Spec,'');
                                                              select '000' INTO stErrorLog;    
   
    SELECT a.tool_id, a.device INTO stTool_id, stDevice
      FROM mcs_rfid_reader_bt a
    WHERE 1=1
      and a.reader_id = stReader_id;
                                                              select '020' INTO stErrorLog;    
  
   SELECT a.cart_id INTO stCart_id 
     FROM mcs_cart_rt a
   WHERE 1=1
     and a.rfid_tag = stRfid_Tag; 
                                                              select '030' INTO stErrorLog;    
   
   START TRANSACTION;
   
   

        IF        ( iIn_Out = 1 ) THEN                                 select '031-00' INTO stErrorLog;    

               IF  STRCMP(stDevice,'LP_1') = 0 THEN                    select '031-01' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id1 = stCart_id WHERE tool_id = stTool_id;

               ELSEIF STRCMP(stDevice,'LP_2') = 0 THEN                 select '031-02' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id2 = stCart_id WHERE tool_id = stTool_id;
      
               ELSEIF STRCMP(stDevice,'LP_3') = 0 THEN                 select '031-03' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id3 = stCart_id WHERE tool_id = stTool_id;      

               ELSEIF STRCMP(stDevice,'LP_4') = 0 THEN                 select '031-04' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id4 = stCart_id WHERE tool_id = stTool_id;
      
               END IF;   
                                                                        select '031-99' INTO stErrorLog;    

       ELSEIF ( iIn_Out = 0 ) THEN                                      select '032-00' INTO stErrorLog;    

               IF  STRCMP(stDevice,'LP_1') = 0 THEN                    select '032-01' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id1 = '', lot_id1 = '' WHERE tool_id = stTool_id;

               ELSEIF STRCMP(stDevice,'LP_2') = 0 THEN                 select '032-02' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id2 = '', lot_id2 = '' WHERE tool_id = stTool_id;
      
               ELSEIF STRCMP(stDevice,'LP_3') = 0 THEN                 select '032-03' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id3 =  '', lot_id3 = '' WHERE tool_id = stTool_id;      

               ELSEIF STRCMP(stDevice,'LP_4') = 0 THEN                 select '032-04' INTO stErrorLog;    

                    UPDATE oee_tool_bt  set cart_id4 =  '', lot_id4 = '' WHERE tool_id = stTool_id;
      
               END IF;   
                                                                       select '032-99' INTO stErrorLog;    

       END IF;   

   
   
   COMMIT;
      
 
 
                                                              select '900' INTO stErrorLog;
  
                                                         
                                                         SELECT 1 into iResult_1_ok_0_ng; 
                                                         SELECT iResult_1_ok_0_ng; 
                                                               call sys_job_exec_sp( 2,stJob_Name,dExec_Time,iElp_Spec,'');


END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `test_array` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8mb4_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `test_array`(
	IN `stArray` VARCHAR(255)
)
BEGIN
DECLARE stRemain TEXT;
DECLARE spilter CHAR(1);
DECLARE pos9 INT DEFAULT 1 ;
DECLARE str9 VARCHAR(1000);

SET spilter = ',';
SET stRemain = stArray;

WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;
   
   IF TRIM(str9) != '' THEN
    -- main porcedure here -------------------------------------------------------
   
	   SELECT str9;
	
    -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;

-- test case -- 2020/03/01 lkchena
-- null: CALL test_array( '' ); 
-- ok1: CALL test_array( 'Test1' ); 
-- ok3: CALL test_array( 'Test1,Test2,Test3' );

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `xls_erp_stb_sp` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `xls_erp_stb_sp`(IN stKey1 varchar(24), IN stKey2 varchar(24), IN stKey3 varchar(24))
    COMMENT 'import xls file stored procedure - 2020/01/02 lkchena'
BEGIN
  

  
  
  
  

  

  DECLARE stJob_Name varchar(32) DEFAULT "xls_erp_stb_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  DECLARE stMfg_No varchar(16) DEFAULT NULL;
  DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;


  DECLARE doneCursor int DEFAULT 0;
  DECLARE cur1 CURSOR FOR
  SELECT a.mfg_no, a.mold_id, a.tool_id
  FROM erp_prod_stb_t9_prod a
    INNER JOIN tool_order_bt b
      ON a.mfg_no = b.mfg_no
      AND a.mold_id = b.mold_id
      AND b.status1 <> 0
      and b.rd_flag = 0 and a.rd_flag = 0; 

                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';
  
  
  

  
  OPEN cur1;
  REPEAT
    FETCH cur1 INTO stMfg_No, stMold_Id, stTool_Id;
    IF NOT doneCursor THEN
      SET stErrorLog = CONCAT('030_', stMfg_No);
      
      DELETE FROM erp_prod_stb_t9_prod
      WHERE mfg_no = stMfg_No
        AND mold_id = stMold_Id 
        and rd_flag = 0;
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CLOSE cur1;
  SET stErrorLog = CONCAT('061_', 'insert compact table');

  
  DELETE FROM tool_order_bt
  WHERE (mfg_no, mold_id) IN (SELECT mfg_no,mold_id FROM erp_prod_stb_t9_prod)
    and rd_flag = 0; 

  SET stErrorLog = '070';

  
  INSERT INTO tool_order_bt (pri, status1, tool_grp_id, tool_id, part_id, part_raw, part_name, part_erp_no, mold_id, powder_type, powder_erp_no, cnt_plan, cnt_act, date_due, date_stb, date_comp, mfg_no, order_no, rd_flag, cust_id, weight, unit, owner_id, owner_name, note, rec_user, rec_time,
                             time_order, ope_no, ope_name, stage_id, stage_name, stage_order) 
    SELECT pri,status1,tool_grp_id,tool_id,part_id,part_raw,part_name,part_erp_no,mold_id,powder_type,powder_erp_no,cnt_plan,cnt_act,date_due,date_stb,date_comp,mfg_no,order_no,rd_flag,cust_id,weight,unit,owner_id,owner_name,note,rec_user,rec_time,
           DATE_FORMAT(NOW(),"%Y/%m/%d %H:00:00") AS time_order,'100.00','成型作業','Forming','成型站', '100' 
          
    FROM erp_prod_stb_t9_prod
  where 1=1
    and rd_flag = 0; 
  
  
  

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `xls_erp_stb_sp_rd` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `xls_erp_stb_sp_rd`(IN stKey1 varchar(24), IN stKey2 varchar(24), IN stKey3 varchar(24))
    COMMENT 'import xls file stored procedure - 2020/01/02 lkchena'
BEGIN
  

  
  
  
  

  DECLARE stJob_Name varchar(32) DEFAULT "xls_erp_stb_sp_rd";
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  DECLARE stMfg_No varchar(16) DEFAULT NULL;
  DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;


  DECLARE doneCursor int DEFAULT 0;
  DECLARE cur1 CURSOR FOR
  SELECT a.mfg_no,a.mold_id, a.tool_id
  FROM erp_prod_stb_t9_rd a
    INNER JOIN tool_order_bt b
      ON a.mfg_no = b.mfg_no
      AND a.mold_id = b.mold_id
      AND b.status1 <> 0
      and b.rd_flag = 1 and a.rd_flag = 1; 

                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';
  
  
  

  
  OPEN cur1;
  REPEAT
    FETCH cur1 INTO stMfg_No, stMold_Id, stTool_Id;
    IF NOT doneCursor THEN
      SET stErrorLog = CONCAT('030_', stMfg_No);
      
      DELETE FROM erp_prod_stb_t9_rd
      WHERE mfg_no = stMfg_No
        AND mold_id = stMold_Id 
        and rd_flag = 1;
    
    END IF;
  UNTIL doneCursor END REPEAT;
  CLOSE cur1;
  SET stErrorLog = CONCAT('061_', 'insert compact table');

  
  DELETE FROM tool_order_bt
  WHERE (mfg_no, mold_id) IN (SELECT mfg_no,mold_id FROM erp_prod_stb_t9_rd)
    and rd_flag = 1; 

  SET stErrorLog = '070';

  
  INSERT INTO tool_order_bt (pri, status1, tool_grp_id, tool_id, part_id, part_raw, part_name, part_erp_no, mold_id, powder_type, powder_erp_no, cnt_plan, cnt_act, date_due, date_stb, date_comp, mfg_no, order_no, rd_flag, cust_id, weight, unit, owner_id, owner_name, note, rec_user, rec_time,
                             time_order, ope_no, ope_name, stage_id, stage_name, stage_order) 
    SELECT pri,status1,tool_grp_id,tool_id,part_id,part_raw,part_name,part_erp_no,mold_id,powder_type,powder_erp_no,cnt_plan,cnt_act,date_due,date_stb,date_comp,mfg_no,order_no,rd_flag,cust_id,weight,unit,owner_id,owner_name,note,rec_user,rec_time,
           DATE_FORMAT(NOW(),"%Y/%m/%d %H:00:00") AS time_order,'100.00','成型作業','Forming','成型站', '100' 
          
    FROM erp_prod_stb_t9_rd
  where 1=1
    and rd_flag = 1; 
  
  
  

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 DROP PROCEDURE IF EXISTS `xxx_order_trackin_box_sp__wait_drop` */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION' */ ;
DELIMITER ;;
CREATE DEFINER=`root`@`localhost` PROCEDURE `xxx_order_trackin_box_sp__wait_drop`(IN stTool_Id varchar(12), stMfg_No varchar(24),stLot_Id varchar(16), stFull_Id varchar(3072),stFull_Time varchar(3072),stFull_2d varchar(3072) )
    COMMENT 'track in box porcedure - 2020/03/01 lkchena'
label_sp: 
BEGIN
-- tip1: order_trackin_box_sp for forming station, others use order_trackin_cart_sp2 -- 2020/03/02 lkchena
-- tip2: we use lot_id act wafer_id in ker_wip_w_bt -- 2020/03/03 lkchena

-- test case: 2020/03/01 lkchena
-- tip: when test, need find the current lot_id -- key: 2020/03/11 lkchena
-- call order_trackin_box_sp('AF35Y01','T-MFG_NO-003','A0100149',',032,048,032',',AB0001,AB0002,AB0003','');
-- call order_trackin_box_sp('AF35Y01','T-MFG_NO-003','A0100146',',032',',AB0001','');
-- call order_trackin_box_sp('AF35Y01','T-MFG_NO-003','A0100148',',048,032',',AB0002,AB0003','');
-- select * from ker_wip_w_bt order by report_time desc
-- SELECT * FROM sys_job_log_bt ORDER BY time_start desc

-- zero record -- 2020/03/04 lkchena
-- delete from ker_wip_w_bt 
-- -- select * from ker_wip_w_bt 
-- where tool_id = 'AF35Y01'
 

  DECLARE stJob_Name varchar(32) DEFAULT "order_trackin_box_sp";
  
  DECLARE iElp_Spec int DEFAULT 30;
  DECLARE dExec_Time datetime DEFAULT NOW();
  DECLARE iResult_1_ok_0_ng int;
  DECLARE stErrorLog varchar(64);
  DECLARE stTmp varchar(255); -- 2020/03/01 lkchena
  

  DECLARE iTmp int DEFAULT 0;
  DECLARE iCNT int DEFAULT 0;
  DECLARE iCumQTY int DEFAULT 0;

  -- DECLARE stTool_Id varchar(12) DEFAULT NULL; 
  -- DECLARE stMfg_No varchar(16) DEFAULT NULL;
  -- DECLARE stMold_Id varchar(16) DEFAULT NULL;

  DECLARE stKeyField varchar(7) DEFAULT NULL; 
  DECLARE stLast1 varchar(19) DEFAULT NULL;
  DECLARE stLast2 varchar(19) DEFAULT NULL;

  -- lot parameter  2020/03/01 lkchena
  DECLARE stHeader  varchar(1) DEFAULT NULL;
  DECLARE stNode_Id varchar(2) DEFAULT NULL; -- 00 ~ zz
  DECLARE iLot_Len int DEFAULT 0;
  DECLARE iLot_Cum int DEFAULT 0;
  
  -- wait drop -- 2020/03/05 lkchena
  -- DECLARE  stArray varchar(255);
  DECLARE stWafer_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  -- DECLARE stLot_Id   varchar(16) DEFAULT NULL; -- 16 ??? 2020/03/01 lkchena
  DECLARE stCart_Id  varchar(16) DEFAULT NULL; -- 2020/03/02 lkchena 
  DECLARE iLot_Size  int DEFAULT 0;  
  DECLARE stRfid_Tag varchar(16) DEFAULT NULL;  

  -- dm use
  DECLARE iRD_Flag      int DEFAULT 0;
  DECLARE stOpe_No      varchar(16) DEFAULT NULL;
  DECLARE stPart_Id     varchar(24) DEFAULT NULL;
  DECLARE stMold_Id     varchar(24) DEFAULT NULL;
  DECLARE stPowder_Type varchar(16) DEFAULT NULL;
  DECLARE stPowder_Id   varchar(24) DEFAULT NULL;
  DECLARE stSTB_Plan    varchar(19) DEFAULT NULL;
  DECLARE stOrder_No    varchar(24) DEFAULT NULL; -- ? ORDER
  
  -- ker use
  DECLARE dREPORT_TIME9 datetime; 
  
  DECLARE stRemain TEXT; -- array use
  DECLARE spilter CHAR(1);
  DECLARE pos9 INT DEFAULT 1 ;
  DECLARE str9 VARCHAR(1000);

  DECLARE doneCursor int DEFAULT 0;

                                             DECLARE CONTINUE HANDLER FOR NOT FOUND SET doneCursor = 1;
                                               DECLARE EXIT HANDLER FOR SQLEXCEPTION
                                               BEGIN
                                                 SELECT  0 INTO iResult_1_ok_0_ng;
                                                 ROLLBACK;
                                                 CALL sys_job_exec_sp(-9, stJob_Name, dExec_Time, iElp_Spec, stErrorLog); 
                                                 SELECT iResult_1_ok_0_ng;
                                               END;                                        
                                               CALL sys_job_exec_sp(1, stJob_Name, dExec_Time, iElp_Spec, '');
                                               SET stErrorLog = '000';					
      -- 0. get variant
      set stWafer_Id = concat(stLot_Id,'.00');
                                               
	  -- 1. delete exist(old) box data                 
       delete from ker_wip_w_bt 
       where lot_id = stWafer_Id; -- stLot_Id; -- 'A0100147'
                                               SET stErrorLog = '001';					
      -- 2. insert real-time data 
      -- tip: claim_time/trackin time use now(), report_time use lot's report_time
	    INSERT INTO ker_wip_w_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, --
                                   cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id, claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                   part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                                   ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func, -- 2020/03/19 lkchena
                                   full_code,full_time,full_2d )
                                  -- different: wafer_id -- 2020/03/05 lkchena
            SELECT report_time,cate, stWafer_Id, lot_id_p,s,lot_size,lot_size_spec,pri,
                   cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id, now() as claim_time, now() as track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                   part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,extra_step,tool_grp_id,tool_grp,proc_time,area_id,
                   ws_type,ws_func,tool_type1,tool_type2,tool_type3,in_out,tool_func, -- 2020/03/19 lkchena
                   -- different
                   stFull_Id,stFull_Time,stFull_2d
             FROM ker_wip_bt
            WHERE lot_id = stLot_Id;
                                                     SET stErrorLog = '011';
      
       
/* ref code, wait drop -- 2020/03/11 lkchena
  -- 1. get lot parameter: 
   SELECT value1 as header,value2 as node_id,ivalue1 as lot_len,ivalue2 as lot_cum 
         into stHeader,stNode_Id,iLot_Len,iLot_Cum
   FROM sys_param_conf_bt where param_id = 'LOT-01'; SET stErrorLog = '001';
   
                               IF stHeader is NULL THEN -- no setting 
                                      SET stErrorLog = 'no sys_param_conf_bt data'; signal sqlstate '45000' set message_text = stErrorLog;
                                      leave label_sp;     
							   END IF;
   -- 1.2 get order base info 
    select  rd_flag,   ope_no,   part_id,   mold_id,   powder_type,   powder_id,   date_stb,   order_no into
		   iRD_Flag, stOpe_No, stPart_Id, stMold_Id, stPowder_Type, stPowder_Id, stSTB_Plan, stOrder_No
      from tool_order_bt 
    where 1=1
     and tool_id = stTool_Id -- 'AF35Y01' / 'T-MFG_NO-003'
     and mfg_no = stMfg_No;                           SET stErrorLog = '003';

   -- 1.8 ker variant
   SET dREPORT_TIME9 = NOW();  
   
   -- qqq: user is not full ready, so we use command part to replace it   
   -- 1.9 use command part
   set stPart_Id = 'ZZZ-01';
   set stMold_Id = 'ZZZ-01';
            
  -- 3. main procedure
  SET spilter = ','; -- array 
  SET stRemain = stArray;

WHILE CHAR_LENGTH(stRemain) > 0 AND pos9 > 0 DO   
   SET pos9 = INSTR(stRemain, spilter);
   IF pos9 = 0 THEN SET str9 = stRemain; ELSE SET str9 = LEFT(stRemain, pos9 - 1); END IF;   
   IF TRIM(str9) != '' THEN
   -- main porcedure here -------------------------------------------------------
   -- ---------------------------------------------------------------------------
                                                         SET stErrorLog = '010';
	  -- 3.1.0
      set stRfid_Tag = str9;
   
      -- 3.1.1 add lot cum and get lot_id
      set iLot_Cum = iLot_Cum + 1;
      set stTmp = LPAD( iLot_Cum, iLot_Len, '0' );
      set stLot_Id = concat(stHeader,stNode_Id,stTmp); -- test:  select stLot_Id;
   
      -- 3.1.2 get cart_id, lot_size / '5060041'
      select a.cart_id, b.cnt_total into stCart_Id, iLot_Size   -- a.cart_type, , b.cnt_box
         from mcs_cart_base_bt a left join mcs_cart_type_bt b on a.cart_type = b.cart_type
       where rfid_tag = stRfid_Tag and active = 1;     SET stErrorLog = '010';

      -- 5.1 insert ker
			 INSERT INTO ker_wip_bt ( report_time,cate,lot_id,lot_id_p,s,lot_size,lot_size_spec,pri, -- 
                                      cart_id,tool_id,order_no,mfg_no,stb_plan,powder_type,powder_id, claim_time, track_in_time,-- lot_size_spec,cast_spec,cast_pcs_spec,
                                      part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,tool_grp_id,tool_grp,proc_time,area_id)
                  SELECT dREPORT_TIME9, 'RT',stLot_Id,stLot_Id,'R' AS s, iLot_Size, iLot_Size, 350, -- 
                        stCart_Id,stTool_Id, stOrder_No, stMfg_No, stSTB_Plan, stPowder_Type, stPowder_Id, now(), now(),-- iLOT_SIZE_SPEC9,iCAST_SPEC9,iCAST_PCS_SPEC9,
                        part_id,part_raw,mold_id,cust_id,ope_no,ope_name,stage_id,stage_name,stage_order,tool_grp_id,tool_grp,proc_time,area_id
                   FROM dm_flow_bt 
                  WHERE part_id = stPart_Id -- 'ZZZ-01' / '100.00'
                    AND  ope_no = stOpe_No;            SET stErrorLog = '011';
   
	  -- FOR DEBUG: SELECT  concat(stPart_Id,' - ',  stLot_Id,' - ', stRfid_Tag,' - ',stCart_Id,' - ',iLot_Size,' - ',stPowder_Type );



     
                                                  SET stErrorLog = '090';
   -- ---------------------------------------------------------------------------
   -- ---------------------------------------------------------------------------
   END IF;
   SET stRemain = SUBSTRING(stRemain, pos9 + 1);   
END WHILE;                                        SET stErrorLog = '091';


   -- 6.1 update lot cum number
	  UPDATE sys_param_conf_bt
         set ivalue2 = iLot_Cum -- + 1, add above
      WHERE param_id = 'LOT-01';

                                                  SET stErrorLog = '099';
*/

                                               SET stErrorLog = '900'; 
                                               SELECT 1 INTO iResult_1_ok_0_ng; 
                                               SELECT iResult_1_ok_0_ng; 
                                               CALL sys_job_exec_sp(9, stJob_Name, dExec_Time, iElp_Spec, '');

END ;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:17:09
